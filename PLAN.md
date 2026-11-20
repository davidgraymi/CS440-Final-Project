# Project status and plan

Handoff notes for picking this project up in a new session. Background, hypothesis and benchmark design are in `README.md`; this file covers what has been done, what we found, and what comes next.

Branch: `trm-sanity-check` (all work so far). Last updated 2026-11-20.

## Status at a glance

| Stage | Status | Where |
|---|---|---|
| TRM implementation matches the official code exactly | Done | `src/cs440_final_project/models.py`, `scripts/verify_against_official.py` |
| Sanity check with a pretrained TRM on Sudoku-Extreme | Done | `results/sanity-check/` |
| Connect Four dataset: 1M positions with exact solver labels | Done (committed) | `data/c4_{train,test}.npz`, `connect4.py` |
| Connect Four training and evaluation | Done | `c4_train.py` (`c4-train`, `c4-eval`, `c4-sweep`) |
| Overnight pilot: TRM vs. ds-only vs. direct | Done | `results/c4-overnight/` (committed from the server) |
| Test-time recursion sweep on the overnight TRM | Implemented, **not yet run** | `uv run c4-sweep` |
| Long training run | Not started; design depends on the sweep | see Next steps |

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

## Next steps

1. **Run the test-time recursion sweep** on the overnight TRM checkpoint (no retraining):
   ```bash
   uv run c4-sweep --ckpt results/c4-overnight/c4-trm-overnight/ckpt.pt
   ```
   This tries T ∈ {1,2,3,4,6} × n ∈ {1,2,3,6,9} for one step each, on 5,000 depth-balanced positions, and plots accuracy against compute for each depth.
   - If deep buckets (d ≥ 9) gain from more recursion and shallow ones don't, the model uses recursion as lookahead, which supports the hypothesis. Then do step 2.
   - If all curves are flat, the positions can be solved by pattern-matching. Then do step 3 first.
   - If accuracy falls away from the trained setting (T=3, n=6), the model only learned that one setting. Step 2 addresses this as well.
2. **Small-step long run:** train TRM with small steps and halting off, e.g. `uv run c4-train --model trm --T 1 --n 2 --no-halt --d-model 256 --batch-size 128`. Then each supervision step is a small unit of thinking, and accuracy by step for each depth directly tests "the step where accuracy jumps grows with d".
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

## Gotchas learned the hard way

- **MPS vs. CPU:** the deep recursion amplifies small floating-point differences. MPS changes about 6% of individual answers, but net accuracy is within noise. Use MPS for development and CPU or CUDA for final numbers.
- **Mac mini toolchain:** Xcode.app's linker can't read the newer Command Line Tools SDK. Build the solver with `DEVELOPER_DIR=/Library/Developer/CommandLineTools make -C third_party/connect4 c4solver`.
- **Building the dataset on the Mac mini starved AdGuard** (DNS timeouts). Build data elsewhere and commit it; the 1M-position dataset is in git.
- **Local network DNS** (192.168.0.253) blocks Hugging Face's CDN (`us.aws.cdn.hf.co`). The Sudoku checkpoint and data were downloaded with `curl --resolve` using a public DNS lookup.
- **Licensing:** Pascal Pons' solver is AGPL-3.0. It stays in `third_party/` (gitignored) and is run as a separate process. The Sudoku checkpoint (`MaximeILL/GAR_ckpts`) states no license, so it isn't committed; download instructions are in the README.
- The official TRM code was run once (with the user's permission) to verify our implementation. It isn't vendored.
