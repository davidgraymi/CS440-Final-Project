# Project status and plan

Handoff notes for picking this project up in a new session. Background, hypothesis and benchmark design are in `README.md`; this file covers what has been done, what we found, and what comes next.

Branch: `c4-small-step-run` (off `main`). Last updated 2026-11-21 (after the Stage 3 go/no-go).

## Status at a glance

| Stage | Status | Where |
|---|---|---|
| TRM implementation matches the official code exactly | Done | `src/cs440_final_project/models.py`, `scripts/verify_against_official.py` |
| Sanity check with a pretrained TRM on Sudoku-Extreme | Done | `results/sanity-check/` |
| Connect Four dataset: 1M positions with exact solver labels | Done (committed) | `data/c4_{train,test}.npz`, `connect4.py` |
| Connect Four training and evaluation | Done | `c4_train.py` (`c4-train`, `c4-eval`, `c4-sweep`) |
| Overnight pilot: TRM vs. ds-only vs. direct | Done | `results/c4-overnight/` (committed from the server) |
| Test-time recursion sweep on the overnight TRM | Done (2026-11-20): flat beyond the trained setting | `results/c4-sweep/` |
| Small-step long run (step 2) | Done: 83.2%; extra steps add ~1 point, not depth-dependent | `results/c4-smallstep/` |
| EXPERIMENT.md Stage 0: r label, search baseline, solver probe | Done | `results/stage0/`, `tests/test_connect4.py` |
| Stage 1: v2 dataset (2.5M solved positions, tactical set by r) | Done (committed) | `data/c4v2_*`, `c4data.py` |
| Stage 2: search and shortcut baselines on the tactical set | Done | `results/c4v2-baselines/` |
| Stage 3: TRM-small seed 0, n_sup 32 | **No-go at 60k steps**; stopped, resumable | `results/c4v2-trm-small-s0/` |

## Write-up: where things stand (2026-11-21)

**Question.** Does a weight-tied recursive model (TRM) use more thinking steps on Connect Four positions that need deeper lookahead?

**What we built.**
- A label for *required search depth* r: the smallest lookahead that proves the best move is better than every alternative. It is checked against a brute-force search in the tests.
- A new dataset of 2.5M solver-labelled positions from solver-played games, split by game with no position overlap. From its test games, a **tactical set** of 12,991 positions has a unique winning move, balanced by r and game stage, so r is only weakly tied to move number (correlation −0.23, down from −0.63).
- Baselines on the tactical set: a depth-k search goes from chance to 100% exactly at k = r, and a "win now, else block" rule solves only r ≤ 2. The set measures lookahead, as intended.
- A guarded, resumable training and data-build setup that runs at background priority beside AdGuard DNS. Over about 12 hours of solver and GPU work, DNS failures stayed at or below AdGuard's own baseline (2.9%), and free memory never fell below 48%.

**What we found.**
1. Pilot TRMs answer in their first supervision step. Extra steps and extra test-time recursion change accuracy by about 1 point, the same at every depth (`results/c4-sweep/`, `results/c4-smallstep/`).
2. On the tactical set, the best pilot model scores 98% at r=1, 94% at r=2, 81% at r=3–4 and about 55% from r=5 on. Accuracy is flat in the number of steps for every r (`results/c4v2-pilot-model/`).
3. The pre-registered run (T=1, n=1, 32 steps, every example runs all 32) learned too slowly. At 60k steps it reached 40% on r=1 and 50% on r=2, below the 60% go/no-go bar, and later steps often undid correct step-1 answers. Diagnostics found no bug and no data problem.

**So far, there is no evidence for the hypothesis.** The models that learn well do it in one step, and the setup built to force multi-step use doesn't learn fast enough on this hardware.

**Next (to decide).**
- A: warm-start the pre-registered run from the pilot model, about 8 h.
- B: from scratch with 16 steps, about 24 h.
- C: write up the current evidence as a negative result.

Details and every number: [EXPERIMENT.md](EXPERIMENT.md), sections 3.5–3.6 and 7.

## Results so far

### Sudoku sanity check (pretrained TRM, no training)

On 2,000 random Sudoku-Extreme test puzzles, a community TRM-MLP checkpoint scores **86.05%** exact accuracy on CPU (85.70% on MPS), against 86.50% reported for it. Harder puzzles need more supervision steps (Spearman 0.29 between solver difficulty and steps to solve), and the learned halting step tracks difficulty (0.31). Details and plot: README, "Sanity check".

### Connect Four overnight pilot (2026-11-19 → 20)

Setup: d_model 256, batch 128, MLP mixer, on the M1 Mac mini (MPS). TRM trained for 339 minutes (17,616 steps). ds-only and direct trained for the same number of steps. Evaluation: 10,000 test positions, 1,000 per depth bucket.

| Model | Params | Layer applications per step | Optimal-move accuracy | Value accuracy |
|---|---|---|---|---|
| **TRM** (n=6, T=3, 2 layers) | ~1.25M | 42 | **89.7%** | 83.3% |
| ds-only (n=0, T=3, 2 layers) | ~1.25M | 6 | 85.8% | 80.2% |
| direct (8 layers, one pass) | ~5.0M | 8 | 84.1% | 79.5% |

Findings:

1. **Recursion helps.** TRM beats the one-pass model by 5.6 points with about 4× fewer parameters. The gap is larger at depth 3–4: TRM 97.6%, ds-only 89.4%, direct 86.7%.
2. **No overfitting.** Training accuracy at the halting step is about 91.2%, against 89.7% on the depth-balanced test set.
3. **Extra supervision steps do nothing.** TRM accuracy is 89.8% after step 1 and 89.7% after step 16. The halting head stops at step 1 (training average 1.1 steps), and depth barely relates to the steps needed (Spearman 0.05). One step already runs 42 layer applications, so the model learned to answer in one step and never learned to use steps 2–16. **The depth hypothesis cannot be seen at supervision-step granularity with this training setup.**

Full numbers: `results/c4-overnight/c4eval-c4-*-overnight/summary.json` and the plots next to them.

### Test-time recursion sweep (2026-11-20)

The overnight TRM, unchanged, run with T ∈ {1,2,3,4,6} × n ∈ {1,2,3,6,9} for one supervision step each, on 5,000 depth-balanced test positions (500 per bucket), on MPS. It took 7 minutes.

| Depth bucket | T=1, n=1 (4 layer applications) | Trained T=3, n=6 (42) | T=6, n=9 (120) |
|---|---|---|---|
| 3–4 | 60.0% | 98.0% | 97.4% |
| 9–12 | 50.4% | 84.6% | 86.2% |
| 17–24 | 54.4% | 81.0% | 79.0% |
| 25–42 | 56.0% | 80.6% | 79.6% |
| All | 62.6% | 89.6% | 89.3% |

**Outcome: the model only learned its trained compute budget.** Below the trained setting, accuracy falls at every depth. Above it, every depth stays flat: 3× more recursion changes results by ±1.5 points, about the noise level at 500 per bucket. Deep positions gain no more than shallow ones. The trained setting reproduces the overnight evaluation (89.6% vs. 89.7%). So test-time recursion gives no evidence for the depth hypothesis with this model, and step 2 is next. Files: `results/c4-sweep/` (`sweep.json`, `acc_vs_compute_by_depth.png`; the run and guard logs sit next to them on the server but `*.log` is gitignored).

## Next steps

**Superseded by [EXPERIMENT.md](EXPERIMENT.md)** (2026-11-20): the full design for the experiment that tests the hypothesis, with fixed success criteria. The items below are kept as history.

**Small-step run result (2026-11-20):** T=1, n=2, no halting, 68.7k steps, best 83.2% optimal-move accuracy (`results/c4-smallstep/`). Extra supervision steps add about 1 point, almost all at step 2, and no more for deep positions, even within one game stage. The pilots also showed the dataset's depth label mostly measures game stage (depth vs. ply correlation −0.63), and 15% of test positions have every legal move optimal.

1. ~~Test-time recursion sweep~~: done, see above.
2. ~~Small-step long run~~ (done, see above): train TRM with small steps and halting off, e.g. `uv run c4-train --model trm --T 1 --n 2 --no-halt --d-model 256 --batch-size 128`. Then each supervision step is a small unit of thinking, and accuracy by step for each depth directly tests "the step where accuracy jumps grows with d".
3. **Harder dataset if needed:** generate positions from near-optimal play (solver-guided moves with noise) instead of safe-random play, so shallow pattern-matching fails more often. This is a change to `connect4.py` sampling.
4. **Compute-matched baseline:** a `direct` model with about as many layers as TRM uses per step (e.g. `--n-layers 42`), so the comparison is matched on compute and not only on training steps.
5. Later: MCTS at matched FLOPs (Tier 1), and the Pgx game suite.

## Running on the Mac mini server

`scripts/run_c4_experiment.sh` runs the whole experiment in the background with safeguards for the services on that machine (AdGuard DNS at 192.168.0.253, Traefik, Uptime Kuma):

```bash
git pull && DNS_STOP=1 nohup scripts/run_c4_experiment.sh > runs/c4-experiment.log 2>&1 &
```

- **Deadline:** TRM gets most of the time before `DEADLINE` (default 07:00). The baselines train for the same number of steps, and the evaluations run last.
- **Resource limits:** lowest CPU priority, 2 cores left free, and PyTorch GPU memory capped at about 4.3 GB (`MPS_MEM_RATIO=0.4`; the low watermark is set to 75% of the cap, since PyTorch requires low ≤ high).
- **Guard:** checks every 30 s. It stops the run after 5 minutes in a row of free memory below 25%, or of DNS failing or slower than 300 ms when `DNS_STOP=1`. It stops immediately below 10% free memory. Log: `runs/c4-guard.log`.
- `SMOKE=1 TAG=smoke D_MODEL=64 BATCH=32` runs a 4-minute end-to-end test. **Run it after any change to the script.**

Speed on the M1 with d_model 256 and batch 128: about 1.0 s per TRM step, and free memory about 33% while training (61% idle).

Training and evaluation run on the GPU (`--device mps`). On Apple silicon the GPU shares the same 16 GB of RAM as AdGuard and everything else, which is why the MPS memory cap matters. The GPU still keeps the CPU cores free for AdGuard.

**Evaluation-only jobs** (sweeps): use `scripts/run_c4_sweep.sh`. It runs at nice 19 with background QoS, 2 threads and an MPS cap of about 1.6 GB. Its guard checks every 10 s and stops the job (SIGTERM, then SIGKILL) if DNS fails 2 checks in a row, is slower than 200 ms 3 checks in a row, the AdGuardHome process is gone, or free memory drops below 40%. It refuses to start unless DNS is healthy and at least 55% of memory is free. The sweep used about 390 MB; free memory stayed at 74% or more and DNS at 29 ms or less.

**Docker Desktop** was limited to 2.5 GB on 2026-11-20 (it was 8 GB, and the VM had grown to fill it, pushing 3 GB into swap). The containers use about 1.3 GB including VM overhead. Backup of the old settings: `~/Library/Group Containers/group.com.docker/settings-store.json.bak-20261120-084237`.

## Gotchas learned the hard way

- **MPS vs. CPU:** the deep recursion amplifies small floating-point differences. MPS changes about 6% of individual answers, but net accuracy is within noise. Use MPS for development and CPU or CUDA for final numbers.
- **Mac mini toolchain:** Xcode.app's linker can't read the newer Command Line Tools SDK. Build the solver with `DEVELOPER_DIR=/Library/Developer/CommandLineTools make -C third_party/connect4 c4solver`.
- **Building the dataset on the Mac mini starved AdGuard** (DNS timeouts). Build data elsewhere and commit it; the 1M-position dataset is in git.
- **Local network DNS** (192.168.0.253) blocks Hugging Face's CDN (`us.aws.cdn.hf.co`). The Sudoku checkpoint and data were downloaded with `curl --resolve` using a public DNS lookup.
- **Licensing:** Pascal Pons' solver is AGPL-3.0. It stays in `third_party/` (gitignored) and is run as a separate process. The Sudoku checkpoint (`MaximeILL/GAR_ckpts`) states no license, so it isn't committed; download instructions are in the README.
- **MPS evaluation at batch 512 hit the 1.6 GB cap** on the second sweep setting: PyTorch's MPS allocator holds on to freed memory across settings. Use batch 128, and `c4-sweep` now empties the MPS cache after each setting. The cap failed safely (PyTorch raised an error; system memory was never touched).
- **Stale console scripts:** `c4-sweep` was added after the last `uv sync`, so `uv run c4-sweep` fails on the server. `run_c4_sweep.sh` calls `sweep_main` through `.venv/bin/python` instead of re-syncing.
- The official TRM code was run once (with the user's permission) to verify our implementation. It isn't vendored.
