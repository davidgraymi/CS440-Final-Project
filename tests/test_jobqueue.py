"""c4-queue on a two-node Ray cluster on one machine: a "mini" head and a
"laptop" worker, each with its own git checkout as its repo (C4_REPO)."""
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

ray = pytest.importorskip("ray")
from ray.cluster_utils import Cluster  # noqa: E402

from cs440_final_project import jobqueue as jq  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GIT = ["git", "-c", "user.name=test", "-c", "user.email=test@example.com"]


def git(repo, *args):
    subprocess.run([*GIT, *args], cwd=repo, check=True, capture_output=True)


def test_pack_unpack_only_replaces_older_files(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "runs/r1").mkdir(parents=True)
    (src / "runs/r1/ckpt.pt").write_bytes(b"new")
    (src / "runs/r1/ckpt.tmp").write_bytes(b"half-written")
    (src / "runs/r1/log.jsonl").write_bytes(b"old-log")
    os.utime(src / "runs/r1/log.jsonl", (1000, 1000))
    (dst / "runs/r1").mkdir(parents=True)
    (dst / "runs/r1/log.jsonl").write_bytes(b"newer-log")  # mtime now: newer than src's

    written = jq.unpack(dst, jq.pack(src, ["runs/r1", "runs/missing"]))
    assert written == ["runs/r1/ckpt.pt"]
    assert (dst / "runs/r1/ckpt.pt").read_bytes() == b"new"
    assert (dst / "runs/r1/log.jsonl").read_bytes() == b"newer-log"
    assert not (dst / "runs/r1/ckpt.tmp").exists()


@pytest.fixture
def make_cluster(tmp_path, monkeypatch):
    """make_cluster(head_slots) -> (queue, head_repo, laptop_repo, cluster, laptop_node)."""
    made = []

    def make(head_slots=1):
        head, laptop = tmp_path / "mini", tmp_path / "laptop"
        (head / "scripts").mkdir(parents=True)
        shutil.copy(ROOT / "scripts/guarded_run.sh", head / "scripts/guarded_run.sh")
        (head / ".gitignore").write_text("runs/\n")
        git(head, "init", "-q")
        git(head, "add", ".")
        git(head, "commit", "-qm", "init")
        git(tmp_path, "clone", "-q", str(head), str(laptop))
        monkeypatch.setenv("RAY_AUTH_MODE", "disabled")
        monkeypatch.setenv("C4_REPO", str(head))  # the submitter sits on the head
        c = Cluster(initialize_head=True, head_node_args={
            "num_cpus": 1, "resources": {jq.HEAD: 1, jq.SLOT: head_slots},
            "env_vars": {"C4_REPO": str(head)}, "include_dashboard": False})
        node = c.add_node(num_cpus=1, resources={jq.SLOT: 1}, env_vars={"C4_REPO": str(laptop)})
        ray.init(address=c.address, namespace=jq.NAMESPACE)
        made.append(c)
        return jq.get_queue(), head, laptop, c, node

    yield make
    ray.shutdown()
    for c in made:
        c.shutdown()


def submit(queue, cmd, **kw):
    kw.setdefault("guard", False)
    return ray.get(queue.submit.remote(jq.make_spec(cmd, **kw)))


def jobs_now(queue):
    return {j["id"]: j for j in ray.get(queue.status.remote())}


def wait_for(cond, timeout=120, what="condition"):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if (v := cond()):
            return v
        time.sleep(0.5)
    raise TimeoutError(what)


def wait_finished(queue, ids, timeout=120):
    def done():
        jobs = jobs_now(queue)
        return jobs if all(jobs[i]["status"] in jq.FINISHED for i in ids) else None
    return wait_for(done, timeout, f"jobs {ids} to finish")


def test_jobs_spread_over_both_nodes_and_results_reach_head(make_cluster):
    queue, head, laptop, *_ = make_cluster(head_slots=1)
    ids = [submit(queue, f"mkdir -p runs/j{i} && echo $C4_REPO > runs/j{i}/out.txt && sleep 2",
                  name=f"j{i}", collect=[f"runs/j{i}"]) for i in range(4)]
    jobs = wait_finished(queue, ids)
    assert all(jobs[i]["status"] == "done" for i in ids), jobs
    ran_on_laptop = [i for i in ids if (laptop / "runs/queue" / f"{i}.log").exists()]
    assert ran_on_laptop and len(ran_on_laptop) < 4, "both nodes should have taken jobs"
    for n, i in enumerate(ids):  # every result and log ends up in the head's repo
        assert (head / f"runs/j{n}/out.txt").exists()
        assert (head / "runs/queue" / f"{i}.log").exists()
    for i in ran_on_laptop:
        assert (head / f"runs/j{ids.index(i)}/out.txt").read_text().strip() == str(laptop)
    assert (head / "runs/queue/jobs.json").exists()


def test_cancel_queued_and_running(make_cluster):
    queue, *_ = make_cluster(head_slots=1)
    long = [submit(queue, "trap 'echo got-term; exit 0' TERM; sleep 300 & wait") for _ in range(3)]
    wait_for(lambda: sum(j["status"] == "running" for j in jobs_now(queue).values()) == 2,
             60, "two jobs running")  # one slot per node: the third waits
    queued = next(i for i, j in jobs_now(queue).items() if j["status"] == "queued")
    for i in long:
        ray.get(queue.cancel.remote(i))
    jobs = wait_finished(queue, long)
    assert all(jobs[i]["status"] == "cancelled" for i in long), jobs
    assert jobs[queued].get("node") is None


def test_run_name_is_collected_and_synced_while_running_and_resumes_from_head(make_cluster):
    queue, head, laptop, *_ = make_cluster(head_slots=0)  # every job runs on the "laptop"
    (head / "runs/r").mkdir(parents=True)
    (head / "runs/r/ckpt.pt").write_text("step-100 from the head")
    cmd = ("cat runs/r/ckpt.pt > runs/r/resumed-from.txt; echo step-200 > runs/r/ckpt.pt; "
           "sleep 6; echo step-300 > runs/r/ckpt.pt; true --run-name r")
    jid = submit(queue, cmd, sync_every=1)
    assert jobs_now(queue)[jid]["collect"] == ["runs/r"]  # from --run-name
    # Mid-run: the head already has the laptop's newer checkpoint.
    wait_for(lambda: (head / "runs/r/ckpt.pt").read_text().strip() == "step-200", 60, "mid-run copy")
    assert jobs_now(queue)[jid]["status"] == "running"
    jobs = wait_finished(queue, [jid])
    assert jobs[jid]["status"] == "done" and jobs[jid]["synced_at"]
    assert (laptop / "runs/r/resumed-from.txt").read_text() == "step-100 from the head"
    assert (head / "runs/r/ckpt.pt").read_text().strip() == "step-300"
    assert (head / "runs/r/resumed-from.txt").exists()


def test_node_dropout_is_an_error_not_a_silent_restart(make_cluster):
    queue, head, laptop, cluster, laptop_node = make_cluster(head_slots=1)
    marker = f"c4-dropout-{os.getpid()}"
    # Keep the head busy so the job lands on the laptop, then drop the laptop.
    blocker = submit(queue, "sleep 300")
    wait_for(lambda: jobs_now(queue)[blocker]["status"] == "running", 60, "blocker running")
    jid = submit(queue, f"mkdir -p runs/d && echo step-1 > runs/d/ckpt.pt && exec -a {marker} sleep 300",
                 collect=["runs/d"], sync_every=1, guard=True)
    wait_for(lambda: jobs_now(queue)[jid].get("synced_at"), 60, "first copy to the head")
    assert jobs_now(queue)[jid]["node_id"] != jobs_now(queue)[blocker]["node_id"]
    cluster.remove_node(laptop_node)
    job = wait_finished(queue, [jid])[jid]
    assert job["status"] == "error" and "last copy on the head" in job["error"], job
    assert (head / "runs/d/ckpt.pt").read_text().strip() == "step-1"
    # The guard stops the job once its queue worker is gone (it checks every 10 s).
    wait_for(lambda: subprocess.run(["pgrep", "-f", marker]).returncode != 0, 60, "orphan job to stop")
    ray.get(queue.cancel.remote(blocker))
    wait_finished(queue, [blocker])  # unguarded: it must be stopped before the cluster goes


def test_mismatched_or_uncommitted_code_is_refused(make_cluster):
    queue, head, laptop, *_ = make_cluster(head_slots=0)
    (laptop / "scripts/extra.sh").write_text("echo hi\n")
    git(laptop, "add", ".")
    git(laptop, "commit", "-qm", "laptop-only change")
    jid = submit(queue, "mkdir -p runs/m && touch runs/m/ran")
    job = wait_finished(queue, [jid])[jid]
    assert job["status"] == "refused" and "submitted at" in job["error"], job
    assert not (laptop / "runs/m/ran").exists()
    # --allow-mismatch runs it anyway.
    jid = submit(queue, "mkdir -p runs/m && touch runs/m/ran", strict=False)
    assert wait_finished(queue, [jid])[jid]["status"] == "done"
    # Uncommitted code on the submitting side: refused at submit time.
    (head / "scripts/wip.sh").write_text("echo wip\n")
    with pytest.raises(ValueError, match="uncommitted"):
        jq.make_spec("echo hi")


def test_guard_runs_and_stops_when_watched_pid_dies():
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
