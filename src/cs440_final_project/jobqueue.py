"""c4-queue: one job queue across both Macs, on a small Ray cluster.

Each Mac joins the cluster with a number of job slots (scripts/ray_node.sh).
A submitted job is a shell command; Ray starts it on whichever Mac has a free
slot, and holds it in the queue until one does. Nothing is started over SSH:
each Mac's Ray node pulls its own work.

    c4-queue submit --name trm-s0 -- \\
        c4-train --model trm --seed 0 --device mps --run-name trm-s0 --resume
    c4-queue status
    c4-queue logs <job-id>          # tail of the job's log, read from whichever Mac ran it
    c4-queue cancel <job-id>        # queued: dropped; running: SIGTERM (checkpoint saved), then SIGKILL

Where things live. Each job runs from the repo root on its Mac (C4_REPO, set by
ray_node.sh) inside scripts/guarded_run.sh (memory guard everywhere, AdGuard
DNS guard on the mini), with output in runs/queue/<job-id>.log there. The head
(the mini) holds the master copy of every run folder:
  - before a job starts on another Mac, the head's copies of its --collect paths
    are brought over (so --resume continues from the latest checkpoint wherever
    the job lands);
  - while it runs, files in those paths that changed are copied back to the head
    every --sync-every seconds (c4-train/c4-ponder rewrite ckpt.pt after every
    evaluation, atomically, so each copy is a complete checkpoint);
  - at the end, the rest (and the job log) follow.
A file is only ever replaced by a NEWER copy, in either direction.
`--run-name X` in the command adds runs/X to --collect automatically.

Failures. Jobs are not retried by default (--retries): if a Mac drops out
mid-run, the job shows as `error` with the time of the last copy on the head,
and resubmitting it with --resume continues from there. Jobs are refused unless
the Mac running them is at the submitter's commit with no uncommitted code
changes on either side (src/, scripts/, pyproject.toml, uv.lock);
--allow-mismatch skips that check for quick experiments.

The queue itself is a detached Ray actor on the head; its record of jobs is
also written to runs/queue/jobs.json there. If the head's Ray is stopped,
queued and running jobs are lost (the record file stays).
"""

import argparse
import io
import json
import os
import platform
import re
import shlex
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath

os.environ.setdefault("RAY_AUTH_MODE", "token")
os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
import ray  # noqa: E402
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy  # noqa: E402

NAMESPACE = "c4"
ACTOR_NAME = "c4-job-queue"
SLOT = "mps"        # custom resource: job slots a Mac offers (ray_node.sh SLOTS)
HEAD = "c4_head"    # custom resource only the head has: pins the queue and result writes there
ON_HEAD = {HEAD: 0.001}
QUEUE_DIR = Path("runs/queue")
CODE_PATHS = ["src", "scripts", "pyproject.toml", "uv.lock"]  # must match across Macs
TERMINATE_GRACE_S = 60  # guarded_run.sh itself waits STOP_GRACE_S (20 s) before SIGKILL
TRANSFER_TIMEOUT_S = 600
FINISHED = ("done", "failed", "refused", "cancelled", "error")


def repo_root() -> Path:
    return Path(os.environ.get("C4_REPO") or Path(__file__).resolve().parents[2])


def _git(repo: Path, *args) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def code_state(repo: Path) -> tuple[str | None, list[str]]:
    """(HEAD commit or None if not a git checkout, uncommitted changes under CODE_PATHS)."""
    head = _git(repo, "rev-parse", "HEAD")
    dirty = _git(repo, "status", "--porcelain", "--", *CODE_PATHS) or ""
    return (head.strip() if head else None), [l for l in dirty.splitlines() if l.strip()]


def code_problem(repo: Path, want: str | None) -> str | None:
    """Why this checkout may not run a job submitted at commit `want`, or None."""
    head, dirty = code_state(repo)
    if head is None:
        return f"{repo} is not a git checkout"
    if want and head != want:
        return f"this Mac's repo is at {head[:9]}, the job was submitted at {want[:9]}"
    if dirty:
        return f"uncommitted code changes in {repo}: " + "; ".join(dirty[:5])
    return None


# ---- moving run folders between Macs ---------------------------------------------------------

def _files(repo: Path, paths) -> dict[str, tuple[int, int]]:
    """{repo-relative file: (mtime_ns, size)} under the given paths, skipping *.tmp
    (half-written checkpoints: the trainers save to .tmp, then rename)."""
    out = {}
    for p in paths:
        root = repo / p
        found = [root] if root.is_file() else (root.rglob("*") if root.is_dir() else [])
        for f in found:
            if f.is_file() and f.suffix != ".tmp":
                st = f.stat()
                out[f.relative_to(repo).as_posix()] = (st.st_mtime_ns, st.st_size)
    return out


def pack(repo: Path, paths) -> bytes:
    """Gzipped tar of the files under the given repo-relative paths (missing ones skipped)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in _files(repo, paths):
            try:
                tar.add(repo / name, arcname=name)
            except FileNotFoundError:  # replaced or removed since listed
                pass
    return buf.getvalue()


def unpack(repo: Path, data: bytes) -> list[str]:
    """Write the tar's files into repo, each only if it is newer than the copy
    already there (by modification time). Atomic per file. Returns those written."""
    written = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for m in tar.getmembers():
            rel = PurePosixPath(m.name)
            if not m.isfile() or rel.is_absolute() or ".." in rel.parts:
                continue
            dest = repo / rel
            if dest.exists() and int(dest.stat().st_mtime) >= m.mtime:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp")
            with os.fdopen(fd, "wb") as out:
                out.write(tar.extractfile(m).read())
            os.utime(tmp, (m.mtime, m.mtime))
            os.replace(tmp, dest)
            written.append(m.name)
    return written


@ray.remote(num_cpus=0)
def _store_results(data: bytes) -> list[str]:
    """On the head: take newer files from a job on another Mac."""
    return unpack(repo_root(), data)


@ray.remote(num_cpus=0)
def _snapshot(paths: list[str]) -> bytes:
    """On the head: its copies of these paths, for a job starting on another Mac."""
    return pack(repo_root(), paths)


@ray.remote(num_cpus=0)
def _tail(path: str, lines: int) -> str:
    p = repo_root() / path
    if not p.exists():
        return f"(no {path} on {platform.node()})"
    return "".join(p.read_text(errors="replace").splitlines(keepends=True)[-lines:])


# ---- running one job ---------------------------------------------------------------------------

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


class _Syncer:
    """Copies a job's changed files to the head (no-op when the job runs on the head)."""

    def __init__(self, repo, paths, queue, jid, log):
        self.repo, self.paths, self.queue, self.jid, self.log = repo, paths, queue, jid, log
        self.sent: dict[str, tuple[int, int]] = {}

    def pull_from_head(self):
        got = unpack(self.repo, ray.get(_snapshot.options(resources=ON_HEAD).remote(self.paths),
                                        timeout=TRANSFER_TIMEOUT_S))
        self.note(f"brought {len(got)} newer file(s) over from the head" if got
                  else "the head has nothing newer for this job")

    def push(self):
        now = _files(self.repo, self.paths)
        changed = [f for f, sig in now.items() if self.sent.get(f) != sig]
        if not changed:
            return
        try:
            data = pack(self.repo, changed)
            ray.get(_store_results.options(resources=ON_HEAD).remote(data), timeout=TRANSFER_TIMEOUT_S)
        except Exception as e:  # the head unreachable for now: try again next time
            self.note(f"copy to the head failed, will retry: {type(e).__name__}: {e}"[:300])
            return
        self.sent.update({f: now[f] for f in changed})
        self.queue.mark_synced.remote(self.jid, time.strftime("%F %T"), len(changed), len(data))

    def note(self, msg):
        self.log.write(f"=== {time.strftime('%F %T')} | {msg}\n")
        self.log.flush()


@ray.remote(num_cpus=0)  # retries are set per job at submit (default 0)
def _run_job(spec: dict) -> dict:
    repo = repo_root()
    node, node_id = platform.node(), ray.get_runtime_context().get_node_id()
    head_commit, _ = code_state(repo)
    queue = ray.get_actor(ACTOR_NAME, namespace=NAMESPACE)
    queue.mark_started.remote(spec["id"], node, node_id, head_commit)

    log_path = repo / QUEUE_DIR / f"{spec['id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log:
        log.write(f"=== {time.strftime('%F %T')} | job {spec['id']} ({spec['name']}) on {node}\n"
                  f"=== repo {repo} at {(head_commit or '?')[:9]} (submitted at {(spec['submit_commit'] or '?')[:9]})\n"
                  f"=== $ {spec['cmd']}\n")
        log.flush()
        if spec["strict"] and (problem := code_problem(repo, spec["submit_commit"])):
            log.write(f"=== REFUSED: {problem}\n")
            return {"rc": None, "refused": problem}

        sync = None
        if not _on_head():
            sync = _Syncer(repo, [*spec["collect"], str(QUEUE_DIR / log_path.name)], queue, spec["id"], log)
            if spec["collect"]:
                sync.pull_from_head()

        argv = ["bash", "-c", spec["cmd"]]
        if spec["guard"]:
            argv = ["bash", str(repo / "scripts/guarded_run.sh"), *argv]
        # The project's venv first on PATH, so `python`, `c4-train` etc. are this repo's.
        env = os.environ | {"C4_JOB_ID": spec["id"], "C4_WATCH_PID": str(os.getpid()),
                            "PATH": f"{repo / '.venv/bin'}{os.pathsep}{os.environ.get('PATH', '')}"}
        proc = subprocess.Popen(argv, cwd=repo, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        try:
            # Poll rather than block: ray.cancel delivers KeyboardInterrupt between
            # Python bytecodes, which never happens inside a blocking proc.wait().
            next_sync = time.time() + spec["sync_every"]
            while (rc := proc.poll()) is None:
                time.sleep(1)
                if sync and time.time() >= next_sync:
                    sync.push()
                    next_sync = time.time() + spec["sync_every"]
        except BaseException:  # ray.cancel raises KeyboardInterrupt here
            _terminate(proc)
            if sync:
                sync.note("cancelled; copying the final state to the head")
                sync.push()
            raise
        log.write(f"=== {time.strftime('%F %T')} | exit code {rc}\n")
        log.flush()
        if sync:
            sync.push()
    return {"rc": rc}


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
                job["rc"] = res["rc"]
                if res.get("refused"):
                    job["status"], job["error"] = "refused", res["refused"]
                else:
                    job["status"] = {0: "done", 75: "refused"}.get(res["rc"], "failed")
            except ray.exceptions.TaskCancelledError:
                job["status"] = "cancelled"
            except Exception as e:  # e.g. its Mac left the cluster mid-run
                last = (f"; last copy on the head from {job['synced_at']}" if job.get("synced_at")
                        else "; nothing was copied to the head" if job.get("node") else "")
                job["status"] = "error"
                job["error"] = f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"[:300] + last
            job["finished_at"] = time.strftime("%F %T")
            del self.refs[jid]
        self._save()

    def submit(self, spec: dict) -> str:
        self.jobs[spec["id"]] = spec | {"status": "queued"}
        self.refs[spec["id"]] = _run_job.options(
            resources={SLOT: spec["slots"]}, max_retries=spec["retries"], name=spec["name"]).remote(spec)
        self._save()
        return spec["id"]

    def mark_started(self, jid, node, node_id, commit):
        self.jobs[jid] |= {"status": "running", "node": node, "node_id": node_id,
                           "node_commit": commit, "started_at": time.strftime("%F %T")}
        self._save()

    def mark_synced(self, jid, when, n_files, n_bytes):
        self.jobs[jid] |= {"synced_at": when, "synced_files": n_files, "synced_bytes": n_bytes}
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


# ---- command line ------------------------------------------------------------------------------

def connect():
    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), namespace=NAMESPACE,
             logging_level="ERROR", log_to_driver=False)


def get_queue():
    return JobQueue.options(name=ACTOR_NAME, namespace=NAMESPACE, lifetime="detached",
                            get_if_exists=True, resources=ON_HEAD).remote()


def make_spec(cmd: str, name=None, collect=(), slots=1.0, guard=True, retries=0,
              strict=True, sync_every=60.0) -> dict:
    """A job record. With strict, raises ValueError if the submitting checkout has
    uncommitted code changes (the job could not be reproduced from a commit)."""
    commit, dirty = code_state(repo_root())
    if strict and (commit is None or dirty):
        raise ValueError(code_problem(repo_root(), None))
    collect = list(collect)
    for run in re.findall(r"--run-name[= ]+(\S+)", cmd):  # c4-train / c4-ponder output folder
        if f"runs/{run}" not in collect:
            collect.append(f"runs/{run}")
    jid = time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    return {"id": jid, "name": name or jid, "cmd": cmd, "collect": collect, "slots": slots,
            "guard": guard, "retries": retries, "strict": strict, "sync_every": sync_every,
            "submitted_at": time.strftime("%F %T"), "submit_commit": commit}


def _print_status(jobs, show_all):
    if not show_all:
        jobs = [j for j in jobs if j["status"] not in FINISHED] + \
               [j for j in jobs if j["status"] in FINISHED][-10:]
    print(f"{'ID':<17} {'NAME':<20} {'STATUS':<10} {'RC':>3}  {'NODE':<14} {'STARTED':<19}  {'SYNCED':<8}  COMMAND")
    for j in sorted(jobs, key=lambda j: j["submitted_at"]):
        rc = "" if j.get("rc") is None else j["rc"]
        print(f"{j['id']:<17} {j['name'][:20]:<20} {j['status']:<10} {rc!s:>3}  "
              f"{(j.get('node') or '')[:14]:<14} {j.get('started_at', ''):<19}  "
              f"{(j.get('synced_at') or '')[11:]:<8}  {j['cmd'][:50]}")
        if j.get("error"):
            print(f"{'':<17} ! {j['error']}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="c4-queue", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)
    s = sub.add_parser("submit", help="queue a shell command (put it after --)")
    s.add_argument("--name", help="short label shown in status (default: the job id)")
    s.add_argument("--collect", action="append", default=[], metavar="PATH",
                   help="repo-relative path kept in sync with the head while the job runs on another "
                        "Mac (repeatable). runs/<X> is added for `--run-name X` in the command")
    s.add_argument("--slots", type=float, default=1.0, help="job slots this job takes (default 1)")
    s.add_argument("--retries", type=int, default=0,
                   help="times to rerun the job (from the start of the command) if its Mac drops "
                        "out (default 0: it shows as an error instead)")
    s.add_argument("--sync-every", type=float, default=60.0, metavar="S",
                   help="seconds between copies of changed files to the head (default 60)")
    s.add_argument("--allow-mismatch", action="store_true",
                   help="run even with uncommitted code changes or a different commit on the Mac")
    s.add_argument("--no-guard", action="store_true",
                   help="run without scripts/guarded_run.sh (then nothing stops the job if its "
                        "Mac's Ray node dies; `c4-queue cancel` still does)")
    s.add_argument("cmd", nargs=argparse.REMAINDER, help="-- command [args...]")
    st = sub.add_parser("status", help="list jobs (unfinished + last 10 finished)")
    st.add_argument("--all", action="store_true", help="list every job")
    lg = sub.add_parser("logs", help="tail a job's log, from whichever Mac it ran on")
    lg.add_argument("id")
    lg.add_argument("-n", type=int, default=40, help="lines (default 40)")
    c = sub.add_parser("cancel", help="drop a queued job or stop a running one")
    c.add_argument("id")
    a = p.parse_args(argv)

    spec = None
    if a.action == "submit":
        cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
        if not cmd:
            p.error("submit: give the command after --")
        try:
            spec = make_spec(cmd[0] if len(cmd) == 1 else shlex.join(cmd), a.name, a.collect, a.slots,
                             not a.no_guard, a.retries, not a.allow_mismatch, a.sync_every)
        except ValueError as e:
            p.error(f"submit: {e} (commit first, or pass --allow-mismatch)")

    connect()
    queue = get_queue()
    if a.action == "submit":
        print(ray.get(queue.submit.remote(spec)))
        if spec["collect"]:
            print(f"kept in sync with the head: {', '.join(spec['collect'])}", file=sys.stderr)
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
        else:  # that Mac left the cluster; the head has the last copy of the log
            print(f"({job.get('node')} is not in the cluster; showing the head's last copy)")
            print(ray.get(_tail.options(resources=ON_HEAD).remote(path, a.n)), end="")
    elif a.action == "cancel":
        print(ray.get(queue.cancel.remote(a.id)))


if __name__ == "__main__":
    main()
