"""EXPERIMENT-OUTER-LOOP.md end to end through the two-Mac job queue (c4-queue).

    nohup .venv/bin/c4-outer-loop >> runs/outer-loop-queue.log 2>&1 &     # on the Mac mini (Ray head)

1. Queues the four pilots. Each job trains (c4-ponder) and then evaluates on the
   full tactical set (c4-ponder-eval), so a job's results travel with it.
2. Waits for them, then applies the section-4 go criteria (c4_ponder.decide_main).
3. If a pilot passes, queues the 3-day phase longest first: main runs (seeds 0
   and 1), then fixed-steps controls K = 24, 8, 4, 1. Each is evaluated on the
   tactical and general sets.

Jobs that end without results (the guard stopped them, a Mac dropped out, the
Mac refused to start) are queued again after --retry-wait seconds, up to
--max-tries times; every training command has --resume, so they continue from
the latest checkpoint the head has. Job ids are kept in runs/queue/outer-loop-state.json,
so rerunning this command after a restart picks up where it left off.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

PILOTS = {  # name suffix -> settings (EXPERIMENT-OUTER-LOOP.md section 4)
    "D": "--window 0 --col-weight 1",  # full credit across steps, per-column target (full design)
    "C": "--window 0 --col-weight 0",  # full credit, move target only
    "B": "--window 1 --col-weight 1",  # no credit across steps, per-column target
    "A": "--window 1 --col-weight 0",  # no credit across steps, move only (control)
}
COMMON = "--T 1 --n 1 --d-model 256 --batch-size 128 --k-max 24 --beta 0.1 --prior-steps 6 --compile --amp fp16"


def log(msg: str) -> None:
    print(f"{time.strftime('%F %T')} | {msg}", flush=True)


class Driver:
    def __init__(self, args):
        from . import jobqueue as jq

        self.jq, self.args = jq, args
        self.repo = jq.repo_root()
        self.state_file = self.repo / jq.QUEUE_DIR / f"{args.prefix}-state.json"
        self.state = json.loads(self.state_file.read_text()) if self.state_file.exists() else {}
        jq.connect()
        self.queue = jq.get_queue()

    def save(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(self.state_file)

    def job_cmd(self, name: str, steps: int, flags: str, evals) -> str:
        a = self.args
        out = f"results/c4p/{name}"
        parts = [f"c4-ponder --run-name {name} --steps {steps} --eval-every {a.eval_every} --device mps "
                 f"--workers 2 --resume {COMMON} {flags}"]
        for split, boot in evals:
            parts.append(f"c4-ponder-eval --ckpt runs/{name}/ckpt.pt --split {split} --bootstrap {boot} "
                         f"--device mps --out {out}/eval-{split}.json" + (f" --limit {a.eval_limit}" if a.eval_limit else ""))
        parts.append(f"cp runs/{name}/log.jsonl runs/{name}/config.json {out}/")
        return " && ".join(parts)

    def done(self, name: str, evals) -> bool:
        return all((self.repo / f"results/c4p/{name}/eval-{split}.json").exists() for split, _ in evals)

    def submit(self, name: str, cmd: str):
        spec = self.jq.make_spec(cmd, name=name, collect=[f"results/c4p/{name}"])
        jid = self.jq.ray.get(self.queue.submit.remote(spec))
        st = self.state.setdefault(name, {"tries": 0})
        st.update(id=jid, tries=st["tries"] + 1, submitted=time.strftime("%F %T"), retry_at=None)
        self.save()
        log(f"{name}: queued as {jid} (try {st['tries']})")

    def run_batch(self, jobs):
        """jobs: list of (name, cmd, evals), in submission order. Returns names that finished with results."""
        for name, cmd, evals in jobs:
            if self.done(name, evals):
                log(f"{name}: results already on the head")
            elif name not in self.state:
                self.submit(name, cmd)  # (after a restart, known jobs are followed, not resubmitted)
        while True:
            status = {j["id"]: j for j in self.jq.ray.get(self.queue.status.remote())}
            pending = 0
            for name, cmd, evals in jobs:
                if self.done(name, evals):
                    continue
                st = self.state[name]
                if st.get("gave_up"):
                    continue
                if st.get("retry_at"):
                    if time.time() >= st["retry_at"]:
                        self.submit(name, cmd)
                    pending += 1
                    continue
                job = status.get(st["id"])
                if job is None:  # the queue lost it (the head's Ray was restarted): queue it again
                    log(f"{name}: job {st['id']} unknown to the queue; queueing again")
                    self.submit(name, cmd)
                    pending += 1
                    continue
                if job["status"] not in self.jq.FINISHED:
                    pending += 1
                    continue
                if job["status"] == "done" and self.done(name, evals):
                    log(f"{name}: done on {job.get('node')}")
                    continue
                why = job.get("error") or f"exit code {job.get('rc')}"
                if st["tries"] >= self.args.max_tries:
                    log(f"{name}: {job['status']} ({why}); giving up after {st['tries']} tries")
                    st["gave_up"] = True
                    self.save()
                    continue
                log(f"{name}: {job['status']} on {job.get('node')} ({why}); retrying in {self.args.retry_wait:.0f} s")
                st["retry_at"] = time.time() + self.args.retry_wait
                self.save()
                pending += 1
            if not pending:
                return [n for n, _, ev in jobs if self.done(n, ev)]
            time.sleep(self.args.poll)

    def run(self):
        a, p = self.args, self.args.prefix
        tactical = [("tactical", 200)]
        log(f"=== {p}: pilots ===")
        pilots = [(f"{p}-pilot-{k}", self.job_cmd(f"{p}-pilot-{k}", a.pilot_steps, f"{v} --seed 0", tactical), tactical)
                  for k, v in PILOTS.items()]
        finished = self.run_batch(pilots)

        decision = self.repo / f"results/c4p/{p}-pilot-decision.json"
        evals = [f"{n}=results/c4p/{n}/eval-tactical.npz" for n in finished]
        out = subprocess.run([str(self.repo / ".venv/bin/c4-ponder-decide"), "--evals", *evals, "--out", str(decision)],
                             cwd=self.repo, capture_output=True, text=True)
        winner = (out.stdout.strip().splitlines() or ["none"])[-1]
        if a.force_winner:
            log(f"pilot decision: {winner} (overridden by --force-winner {a.force_winner})")
            winner = f"{p}-pilot-{a.force_winner}"
        log(f"pilot decision: winner = {winner} (details: {decision.relative_to(self.repo)})")
        if winner == "none":
            log("no pilot passed the go criteria; no multi-day run")
            return
        flags = PILOTS[winner.rsplit("-", 1)[1]]
        log(f"=== {p}: 3-day phase with {winner} settings ({flags}) ===")
        both = [("tactical", 1000), ("general", 200)]
        jobs = [(f"{p}-main-s{s}", self.job_cmd(f"{p}-main-s{s}", a.main_steps, f"{flags} --seed {s}", both), both)
                for s in (0, 1)]
        jobs += [(f"{p}-fixed-k{k}", self.job_cmd(f"{p}-fixed-k{k}", a.main_steps, f"{flags} --fixed-steps {k} --seed 0",
                                                  both), both) for k in (24, 8, 4, 1)]  # longest first
        self.run_batch(jobs)
        log(f"=== {p}: all done ===")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="c4p", help="run-name prefix (use another for dry runs)")
    ap.add_argument("--pilot-steps", type=int, default=2200)
    ap.add_argument("--main-steps", type=int, default=16000)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-limit", type=int, default=0, help="dry runs: evaluate only this many positions")
    ap.add_argument("--max-tries", type=int, default=8)
    ap.add_argument("--retry-wait", type=float, default=600)
    ap.add_argument("--poll", type=float, default=60)
    ap.add_argument("--force-winner", choices=list(PILOTS), help="dry runs: skip the decision")
    Driver(ap.parse_args(argv)).run()


if __name__ == "__main__":
    main()
