"""c4-queue on a two-node Ray cluster on one machine: a "mini" head and a
"laptop" worker, each with its own repo directory (C4_REPO)."""
import os
import subprocess
import time
from pathlib import Path

import pytest

ray = pytest.importorskip("ray")
from ray.cluster_utils import Cluster  # noqa: E402

from cs440_final_project import jobqueue as jq  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_pack_unpack_roundtrip(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "runs/r1").mkdir(parents=True)
    (src / "runs/r1/ckpt.pt").write_bytes(b"x" * 1000)
    names = jq.unpack(dst, jq.pack(src, ["runs/r1", "runs/missing"]))
    assert "runs/r1/ckpt.pt" in names
    assert (dst / "runs/r1/ckpt.pt").read_bytes() == b"x" * 1000


@pytest.fixture
def cluster(tmp_path, monkeypatch):
    head_repo, laptop_repo = tmp_path / "mini", tmp_path / "laptop"
    for r in (head_repo, laptop_repo):
        (r / "scripts").mkdir(parents=True)
        (r / "scripts/guarded_run.sh").symlink_to(ROOT / "scripts/guarded_run.sh")
    monkeypatch.setenv("RAY_AUTH_MODE", "disabled")
    monkeypatch.setenv("C4_REPO", str(head_repo))  # the driver (submitter) sits on the head
    c = Cluster(initialize_head=True, head_node_args={
        "num_cpus": 1, "resources": {jq.HEAD: 1, jq.SLOT: 1},
        "env_vars": {"C4_REPO": str(head_repo)}, "include_dashboard": False})
    c.add_node(num_cpus=1, resources={jq.SLOT: 1}, env_vars={"C4_REPO": str(laptop_repo)})
    ray.init(address=c.address, namespace=jq.NAMESPACE)
    yield jq.get_queue(), head_repo, laptop_repo
    ray.shutdown()
    c.shutdown()


def wait_finished(queue, ids, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        jobs = {j["id"]: j for j in ray.get(queue.status.remote())}
        if all(jobs[i]["status"] in jq.FINISHED for i in ids):
            return jobs
        time.sleep(0.5)
    raise TimeoutError(jobs)


def test_jobs_spread_over_both_nodes_and_results_reach_head(cluster):
    queue, head, laptop = cluster
    ids = [ray.get(queue.submit.remote(jq.make_spec(
        f"mkdir -p runs/j{i} && echo $C4_REPO > runs/j{i}/out.txt && sleep 2",
        name=f"j{i}", collect=[f"runs/j{i}"], guard=False))) for i in range(4)]
    jobs = wait_finished(queue, ids)
    assert all(jobs[i]["status"] == "done" for i in ids)
    ran_on_laptop = [i for i in ids if (laptop / "runs/queue" / f"{i}.log").exists()]
    assert ran_on_laptop and len(ran_on_laptop) < 4, "both nodes should have taken jobs"
    for n, i in enumerate(ids):  # every result and log ends up in the head's repo
        assert (head / f"runs/j{n}/out.txt").exists()
        assert (head / "runs/queue" / f"{i}.log").exists()
    for i in ran_on_laptop:
        n = ids.index(i)
        assert (head / f"runs/j{n}/out.txt").read_text().strip() == str(laptop)
    assert (head / "runs/queue/jobs.json").exists()


def test_cancel_queued_and_running(cluster):
    queue, head, _ = cluster
    long = [ray.get(queue.submit.remote(jq.make_spec(
        "trap 'echo got-term; exit 0' TERM; sleep 300 & wait", guard=False))) for _ in range(3)]
    t0 = time.time()  # two run (one slot per node), the third waits
    while sum(j["status"] == "running" for j in ray.get(queue.status.remote())) < 2:
        assert time.time() - t0 < 60
        time.sleep(0.5)
    queued = next(j["id"] for j in ray.get(queue.status.remote()) if j["status"] == "queued")
    for i in long:
        ray.get(queue.cancel.remote(i))
    jobs = wait_finished(queue, long)
    assert all(jobs[i]["status"] == "cancelled" for i in long), jobs
    assert jobs[queued].get("node") is None


def test_guard_runs_and_stops_when_watched_pid_dies(tmp_path):
    ok = subprocess.run(["bash", ROOT / "scripts/guarded_run.sh", "echo", "hello"],
                        capture_output=True, text=True, env=os.environ | {"C4_GUARD_DNS": "0"})
    if ok.returncode == 75:
        pytest.skip("not enough free memory for the guard to start a job")
    assert ok.returncode == 0 and "hello" in ok.stdout

    watched = subprocess.Popen(["sleep", "300"])
    guard = subprocess.Popen(["bash", ROOT / "scripts/guarded_run.sh", "sleep", "300"],
                             stdout=subprocess.PIPE, text=True,
                             env=os.environ | {"C4_GUARD_DNS": "0", "C4_WATCH_PID": str(watched.pid),
                                               "STOP_GRACE_S": "2"})
    time.sleep(2)
    watched.kill()
    watched.wait()
    out, _ = guard.communicate(timeout=30)
    assert guard.returncode == 2
    assert "queue worker" in out
