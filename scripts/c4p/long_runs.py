"""Longer pilots (10x): D (full design), A (control), C (full credit, move only).
Fresh runs, one cosine schedule over 22k updates. Reuses outer_loop.Driver's
queue/retry logic; state in runs/queue/c4p-long-state.json. Run from the repo root.

v2 (2026-11-23): max_tries 50; also evaluates the four 2,200-update pilots on the
general set, so every c4p model has both tactical and general results."""
import argparse
from cs440_final_project.outer_loop import Driver, PILOTS, log

STEPS = 22000


class EvalDriver(Driver):
    def submit(self, name, cmd):
        if not name.startswith("c4p-pilot-"):
            return super().submit(name, cmd)
        # eval-only job: its Mac also needs the checkpoint, so bring runs/<name> along
        import time
        spec = self.jq.make_spec(cmd, name=f"{name}-eval", collect=[f"results/c4p/{name}", f"runs/{name}"])
        jid = self.jq.ray.get(self.queue.submit.remote(spec))
        st = self.state.setdefault(name, {"tries": 0})
        st.update(id=jid, tries=st["tries"] + 1, submitted=time.strftime("%F %T"), retry_at=None)
        self.save()
        log(f"{name}: eval queued as {jid} (try {st['tries']})")


args = argparse.Namespace(prefix="c4p-long", max_tries=50, retry_wait=300, poll=60,
                          eval_every=1000, eval_limit=0, force_winner=None)
d = EvalDriver(args)
both = [("tactical", 1000), ("general", 200)]
jobs = [(f"c4p-long-{k}", d.job_cmd(f"c4p-long-{k}", STEPS, f"{PILOTS[k]} --seed 0", both), both)
        for k in ("D", "A", "C")]
gen = [("general", 200)]
jobs += [(f"c4p-pilot-{k}",
          f"c4-ponder-eval --ckpt runs/c4p-pilot-{k}/ckpt.pt --split general --bootstrap 200 --device mps "
          f"--out results/c4p/c4p-pilot-{k}/eval-general.json", gen) for k in ("D", "C", "B", "A")]
log(f"=== long pilots + pilot general evals: {[j[0] for j in jobs]}, max tries {args.max_tries} ===")
d.run_batch(jobs)
log("=== long pilots: all done ===")
