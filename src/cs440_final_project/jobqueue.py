"""c4-queue: one job queue across both Macs, on a small Ray cluster.

Each Mac joins the cluster with a number of job slots (scripts/ray_node.sh).
A submitted job is a shell command; Ray starts it on whichever Mac has a free
slot, and holds it in the queue until one does. Nothing is started over SSH:
each Mac's Ray node pulls its own work.

    c4-queue submit --name trm-s0 --collect runs/trm-s0 -- \\
        c4-train --model trm --seed 0 --device mps --run-name trm-s0 --resume
    c4-queue status
    c4-queue logs <job-id>          # tail of the job's log, read from whichever Mac ran it
    c4-queue cancel <job-id>        # queued: dropped; running: SIGTERM (checkpoint saved), then SIGKILL

Each job runs from the repo root on its Mac (C4_REPO, set by ray_node.sh) inside
scripts/guarded_run.sh, which applies the same memory guard as run_c4_train.sh
and, on the Mac mini, the AdGuard DNS guard. Its output goes to
runs/queue/<job-id>.log on that Mac. When a job finishes on a Mac other than the
head (the mini), its log and the --collect paths are copied to the same paths
in the head's repo, through Ray, so all results end up on the mini.

The queue itself is a detached Ray actor on the head; its record of jobs is
also written to runs/queue/jobs.json there. If the head's Ray is stopped,
queued and running jobs are lost (the record file stays).
"""

import argparse
import io
import json
import os
import platform
import shlex
import signal
import subprocess
import sys
import tarfile
import time
import uuid
from pathlib import Path

os.environ.setdefault("RAY_AUTH_MODE", "token")
os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
import ray  # noqa: E402
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy  # noqa: E402

NAMESPACE = "c4"
ACTOR_NAME = "c4-job-queue"
SLOT = "mps"        # custom resource: job slots a Mac offers (ray_node.sh SLOTS)
HEAD = "c4_head"    # custom resource only the head has: pins the queue and result writes there
QUEUE_DIR = Path("runs/queue")
TERMINATE_GRACE_S = 60  # guarded_run.sh itself waits STOP_GRACE_S (20 s) before SIGKILL
FINISHED = ("done", "failed", "refused", "cancelled", "error")


def repo_root() -> Path:
    return Path(os.environ.get("C4_REPO") or Path(__file__).resolve().parents[2])


def git_head(repo: Path) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "?"
    except (OSError, subprocess.SubprocessError):
        return "?"


def pack(repo: Path, paths) -> bytes:
    """Gzipped tar of the given repo-relative paths (missing ones are skipped)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in paths:
            if (repo / p).exists():
                tar.add(repo / p, arcname=str(p))
    return buf.getvalue()


def unpack(repo: Path, data: bytes) -> list[str]:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        names = tar.getnames()
        tar.extractall(repo, filter="data")  # "data" refuses absolute paths and ../ escapes
    return names


def _on_head() -> bool:
    me = ray.get_runtime_context().get_node_id()
    return any(n["NodeID"] == me and HEAD in n["Resources"] for n in ray.nodes() if n["Alive"])


def _terminate(proc: subprocess.Popen) -> None:
    """SIGTERM the job's wrapper (it stops the job gracefully), SIGKILL if it hangs."""
    for sig, wait_s in ((signal.SIGTERM, TERMINATE_GRACE_S), (signal.SIGKILL, 10)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(wait_s)
            return
        except subprocess.TimeoutExpired:
            pass


@ray.remote(num_cpus=0)
def _store_results(data: bytes) -> list[str]:
    """Runs on the head: unpack a finished job's results into its repo."""
    return unpack(repo_root(), data)


@ray.remote(num_cpus=0)
def _tail(path: str, lines: int) -> str:
    p = repo_root() / path
    if not p.exists():
        return f"(no {path} on {platform.node()})"
    return "".join(p.read_text(errors="replace").splitlines(keepends=True)[-lines:])


@ray.remote(num_cpus=0, max_retries=2)  # retried (from the start) only if its Mac drops out
def _run_job(spec: dict) -> dict:
    repo = repo_root()
    node, node_id, commit = platform.node(), ray.get_runtime_context().get_node_id(), git_head(repo)
    queue = ray.get_actor(ACTOR_NAME, namespace=NAMESPACE)
    queue.mark_started.remote(spec["id"], node, node_id, commit)

    log = repo / QUEUE_DIR / f"{spec['id']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    argv = ["bash", "-c", spec["cmd"]]
    if spec["guard"]:
        argv = ["bash", str(repo / "scripts/guarded_run.sh"), *argv]
    # The project's venv first on PATH, so `python`, `c4-train` etc. are this repo's.
    env = os.environ | {"C4_JOB_ID": spec["id"], "C4_WATCH_PID": str(os.getpid()),
                        "PATH": f"{repo / '.venv/bin'}{os.pathsep}{os.environ.get('PATH', '')}"}
    with open(log, "a") as f:
        f.write(f"=== {time.strftime('%F %T')} | job {spec['id']} ({spec['name']}) on {node}\n"
                f"=== repo {repo} at {commit} (submitted from {spec['submit_commit']})\n"
                f"=== $ {spec['cmd']}\n")
        if commit != spec["submit_commit"]:
            f.write("=== WARNING: this Mac's repo is at a different commit than the submitter's\n")
        f.flush()
        proc = subprocess.Popen(argv, cwd=repo, stdout=f, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        try:
            # Poll rather than block: ray.cancel delivers KeyboardInterrupt between
            # Python bytecodes, which never happens inside a blocking proc.wait().
            while (rc := proc.poll()) is None:
                time.sleep(1)
        except BaseException:  # ray.cancel raises KeyboardInterrupt here
            _terminate(proc)
            raise
        f.write(f"=== {time.strftime('%F %T')} | exit code {rc}\n")

    copied = []
    if not _on_head():  # the job log always comes back; --collect paths with it
        paths = [*spec["collect"], str(QUEUE_DIR / log.name)]
        copied = ray.get(_store_results.options(resources={HEAD: 0.001}).remote(pack(repo, paths)))
    return {"rc": rc, "copied": len(copied)}


@ray.remote(num_cpus=0)
class JobQueue:
    """Keeps the job records and owns the job tasks, so they outlive `c4-queue submit`."""

    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self.refs: dict[str, ray.ObjectRef] = {}
        self.file = repo_root() / QUEUE_DIR / "jobs.json"
        if self.file.exists():  # keep the history of an earlier head session (unfinished ones are gone)
            for j in json.loads(self.file.read_text()):
                if j["status"] not in FINISHED:
                    j["status"], j["error"] = "error", "lost: the head's Ray was restarted"
                self.jobs[j["id"]] = j

    def _save(self):
        self.file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(list(self.jobs.values()), indent=1))
        tmp.replace(self.file)

    def _refresh(self):
        for jid, ref in list(self.refs.items()):
            if not ray.wait([ref], timeout=0)[0]:
                continue
            job = self.jobs[jid]
            try:
                res = ray.get(ref)
                job["rc"], job["copied"] = res["rc"], res["copied"]
                job["status"] = {0: "done", 75: "refused"}.get(res["rc"], "failed")
            except ray.exceptions.TaskCancelledError:
                job["status"] = "cancelled"
            except Exception as e:  # node lost after retries, actor errors, ...
                job["status"], job["error"] = "error", f"{type(e).__name__}: {e}"[:500]
            job["finished_at"] = time.strftime("%F %T")
            del self.refs[jid]
        self._save()

    def submit(self, spec: dict) -> str:
        self.jobs[spec["id"]] = spec | {"status": "queued"}
        self.refs[spec["id"]] = _run_job.options(
            resources={SLOT: spec["slots"]}, name=spec["name"]).remote(spec)
        self._save()
        return spec["id"]

    def mark_started(self, jid, node, node_id, commit):
        self.jobs[jid] |= {"status": "running", "node": node, "node_id": node_id,
                           "node_commit": commit, "started_at": time.strftime("%F %T")}
        self._save()

    def status(self) -> list[dict]:
        self._refresh()
        return list(self.jobs.values())

    def cancel(self, jid: str) -> str:
        if jid not in self.refs:
            return f"{jid}: not queued or running"
        ray.cancel(self.refs[jid])
        self.jobs[jid]["status"] = "cancelling"
        self._save()
        return f"{jid}: cancelling"


def connect():
    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), namespace=NAMESPACE,
             logging_level="ERROR", log_to_driver=False)


def get_queue():
    return JobQueue.options(name=ACTOR_NAME, namespace=NAMESPACE, lifetime="detached",
                            get_if_exists=True, resources={HEAD: 0.001}).remote()


def make_spec(cmd: str, name=None, collect=(), slots=1.0, guard=True) -> dict:
    jid = time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    return {"id": jid, "name": name or jid, "cmd": cmd, "collect": list(collect),
            "slots": slots, "guard": guard, "submitted_at": time.strftime("%F %T"),
            "submit_commit": git_head(repo_root())}


def _print_status(jobs, show_all):
    if not show_all:
        jobs = [j for j in jobs if j["status"] not in FINISHED] + \
               [j for j in jobs if j["status"] in FINISHED][-10:]
    print(f"{'ID':<17} {'NAME':<22} {'STATUS':<10} {'RC':>3}  {'NODE':<16} {'STARTED':<19}  COMMAND")
    for j in sorted(jobs, key=lambda j: j["submitted_at"]):
        rc = "" if j.get("rc") is None else j["rc"]
        print(f"{j['id']:<17} {j['name'][:22]:<22} {j['status']:<10} {rc!s:>3}  "
              f"{j.get('node', '')[:16]:<16} {j.get('started_at', ''):<19}  {j['cmd'][:60]}")
        if j.get("error"):
            print(f"{'':<17} ! {j['error']}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="c4-queue", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)
    s = sub.add_parser("submit", help="queue a shell command (put it after --)")
    s.add_argument("--name", help="short label shown in status (default: the job id)")
    s.add_argument("--collect", action="append", default=[], metavar="PATH",
                   help="repo-relative path to copy to the head Mac when the job ends "
                        "elsewhere (repeatable), e.g. runs/<run-name>")
    s.add_argument("--slots", type=float, default=1.0, help="job slots this job takes (default 1)")
    s.add_argument("--no-guard", action="store_true", help="run without scripts/guarded_run.sh")
    s.add_argument("cmd", nargs=argparse.REMAINDER, help="-- command [args...]")
    st = sub.add_parser("status", help="list jobs (unfinished + last 10 finished)")
    st.add_argument("--all", action="store_true", help="list every job")
    lg = sub.add_parser("logs", help="tail a job's log, from whichever Mac it ran on")
    lg.add_argument("id")
    lg.add_argument("-n", type=int, default=40, help="lines (default 40)")
    c = sub.add_parser("cancel", help="drop a queued job or stop a running one")
    c.add_argument("id")
    a = p.parse_args(argv)

    connect()
    queue = get_queue()
    if a.action == "submit":
        cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
        if not cmd:
            p.error("submit: give the command after --")
        spec = make_spec(cmd[0] if len(cmd) == 1 else shlex.join(cmd), a.name, a.collect,
                         a.slots, not a.no_guard)
        print(ray.get(queue.submit.remote(spec)))
    elif a.action == "status":
        _print_status(ray.get(queue.status.remote()), a.all)
    elif a.action == "logs":
        job = next((j for j in ray.get(queue.status.remote()) if j["id"] == a.id), None)
        if job is None:
            sys.exit(f"no job {a.id}")
        if not job.get("node_id"):
            sys.exit(f"{a.id} has not started yet ({job['status']})")
        alive = {n["NodeID"] for n in ray.nodes() if n["Alive"]}
        path = str(QUEUE_DIR / f"{a.id}.log")
        if job["node_id"] in alive:
            where = NodeAffinitySchedulingStrategy(job["node_id"], soft=False)
            print(ray.get(_tail.options(scheduling_strategy=where).remote(path, a.n)), end="")
        else:  # that Mac left the cluster; the head has the log if it was collected
            print(f"({job.get('node')} is not in the cluster; showing the head's copy)")
            print(ray.get(_tail.options(resources={HEAD: 0.001}).remote(path, a.n)), end="")
    elif a.action == "cancel":
        print(ray.get(queue.cancel.remote(a.id)))


if __name__ == "__main__":
    main()
