"""3-day phase of EXPERIMENT-OUTER-LOOP.md section 5, after long-D passed the go criteria
(2026-11-23). long-D (seed 0, 22k updates) serves as main seed 0; this adds seed 1 and the
fixed-steps controls K = 24, 8, 4, 1, all with D's settings and 22k updates (plan said 16k;
22k matches long-D). Longest first. State in runs/queue/c4p-main-state.json."""
import argparse
from cs440_final_project.outer_loop import Driver, PILOTS, log

STEPS = 22000
args = argparse.Namespace(prefix="c4p-main", max_tries=50, retry_wait=300, poll=60,
                          eval_every=1000, eval_limit=0, force_winner=None)
d = Driver(args)
both = [("tactical", 1000), ("general", 200)]
D = PILOTS["D"]
jobs = [("c4p-main-s1", d.job_cmd("c4p-main-s1", STEPS, f"{D} --seed 1", both), both)]
jobs += [(f"c4p-fixed-k{k}", d.job_cmd(f"c4p-fixed-k{k}", STEPS, f"{D} --fixed-steps {k} --seed 0", both), both)
         for k in (24, 8, 4, 1)]
log(f"=== main phase: {[j[0] for j in jobs]}, {STEPS} updates each ===")
d.run_batch(jobs)
log("=== main phase: all done ===")
