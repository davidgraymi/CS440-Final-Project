# Experiment design: does a recursive model's thinking track lookahead depth?

This is the full design for the experiment that tests the project's hypothesis, written after the pilot runs (overnight TRM, test-time sweep, small-step run; see `PLAN.md`). It fixes the problems those runs exposed, and fixes the success criteria **before** running anything, so the result can't be argued into a success afterwards.

## 1. The claim, made testable

> A weight-tied recursive model does something like search in latent space: the number of thinking iterations it needs to find the right move grows with the lookahead depth that the position requires.

**Primary prediction.** On positions that need search depth *r* to find the winning move, accuracy as a function of thinking iterations *K* stays low for small *K* and rises once *K* is large enough. The *K* where it rises, called *K\*(r)*, increases with *r*.

**What would refute it.**
- *K\*(r)* is the same for every *r*: the model answers in a fixed number of iterations, or not at all.
- Deep positions are solved at the same *K* as shallow ones: pattern-matching, not search.

## 2. What went wrong in the pilots (each one shaped this design)

| Pilot finding | Cause | Fix in this design |
|---|---|---|
| Deep positions (d 25–42) are early-game positions (about 10 plies played); depth and ply correlate at −0.63 | `d` = plies until the game ends under perfect play. For early positions that is long theory, not lookahead anyone computes | New label **r = required search depth** (section 3.2); evaluation stratified by game stage |
| 15% of test positions have every legal move optimal (all of bucket d=2, 29% of draws) | Positions where any move counts as correct | Evaluation uses only positions with a real choice, mostly a **unique** winning move |
| Positions come from random play | Critical moves are often visible tactics | Positions from engine and near-optimal play (3.1) |
| Steps 2–16 add about 1 point, at every depth | One step (6–42 layer applications) already does most of the work, and the labels don't separate depths | Smaller steps, more of them (*K* up to 32), and the r-label |
| No yardstick for what "search-like" accuracy curves look like | | **Search baseline**: depth-limited alpha-beta on the same positions (5.2) |
| Train/test overlap not checked for mirror images, though training mirrors boards | | Deduplicate by canonical (mirror-min) position across splits |
| Constant learning rate after warmup | | Cosine decay |

## 3. Dataset

### 3.1 Positions

Sample unique, non-terminal positions from three kinds of play, so positions resemble real games and not random noise:

| Source | Share | How |
|---|---|---|
| Engine play | 50% | Both sides use a depth-4 alpha-beta with a threat-count heuristic, with ε = 0.15 random moves |
| Near-optimal play | 25% | Moves chosen from the solver's optimal moves, with ε = 0.25 random moves (costs one solve per ply, so a smaller share) |
| Safe-random play | 25% | Current generator (keeps continuity with the pilots) |

Stop each game at a uniformly random ply between 6 and 36. Store the **move string** and the solver's **score for every column** (the pilots kept only the optimal-move mask, which loses information).

**Splits.** Deduplicate by canonical form, the smaller of the position and its mirror image. Split by canonical form, so no test position or its mirror appears in training. About 2M training positions. That's more than enough: at batch 128 with 32 supervision steps per example, 100k optimizer steps consume about 400k examples.

### 3.2 Labels

From the per-column scores (no extra solving):

- **Optimal moves, value, d**: as in the pilots.
- **r, required search depth** (new). A depth-*k* search with no evaluation function can choose correctly if it can either prove the winning move (*k* ≥ the win distance *d*) or refute every alternative (*k* ≥ the loss distance of every other move). So for a won position:
  `r = min(d, max over non-winning moves of their loss distance)`, and `r = d` if any alternative draws or wins.
  This separates positions where the win is deep but the alternatives lose immediately (shallow *r*, solvable by elimination) from positions that genuinely need lookahead. Unit tests check *r* against brute-force depth-limited search on small positions, as `test_connect4.py` does for *d*.

### 3.3 Evaluation sets

1. **Tactical set (primary):** positions where the side to move wins, has **exactly one winning move**, and has *r* in {1, 3, 5, 7, 9, 11, 13, ≥15}.
   - 1,500 positions per *r* bucket, 12,000 in all.
   - Within each bucket, equal numbers from three game stages (plies 6–16, 17–26, 27–36) where possible. Counts per cell are reported, so stage can be controlled in the analysis.
   - Chance accuracy is 1/(legal moves), about 15–20%.
2. **General set:** 10,000 positions balanced by *d*, as in the pilots, for continuity.
3. **Pons' benchmark** (Begin/Middle/End × Easy/Medium/Hard), for comparison with published work.

### 3.4 Where to build it

The solver run is the risky step for the home network. It starved AdGuard last time, with 6 solver processes on 6 of 8 cores.

- **Stage 0 (on this machine, 10 minutes, guarded):** measure one solver process at background priority, which keeps it on the efficiency cores, away from AdGuard:
  - positions per second
  - memory: about 120 MB per process expected (80 MB lookup table + 32 MB opening book)
  - DNS failure rate against the 3% baseline
- **Decide from the numbers:**
  - If 1–2 background workers leave DNS at baseline and finish in under about 24 hours, build here with a guarded, sharded, resumable runner, like `run_c4_train.sh`, writing shards so a guard stop loses nothing.
  - Otherwise build on a laptop or a cloud VM. The pilot dataset took about 12 minutes per 1M positions on an 8-core laptop.

### 3.5 Stage 0 results (2026-11-20)

**Solver on this machine** (safe-random positions at plies 6–36, nice 19 + background QoS, DNS probed every 2 s):

| Workers | Positions/s | Memory | DNS failures | Free memory |
|---|---|---|---|---|
| 1 | 48 | 113 MB | 2 of 221 (0.9%) | ≥ 72% |
| 2 | 101 | 227 MB | 1 of 88 (1.1%) | ≥ 73% |

Both are below AdGuard's own baseline failure rate (2.9%, measured with nothing running). Positions at plies 12–16 are the slowest (mean 80 ms, max 1.7 s), because they fall just past the opening book. **The dataset can be built here with 2 background workers.**

**r label:** `required_depth` in `connect4.py`. It matches brute-force depth-limited search on every won test position (`test_required_depth_matches_search`). The search baseline, `search_baseline_accuracy`, is computed exactly from the per-column scores, so it needs no search code. The test checks it is certain to be right exactly from k = r on.

**Yield** (6,000 safe-random positions): 58% are won, and 35% are won with a unique winning move.

| r | Unique-win positions per 6,000 | Plies 6–16 | 17–26 | 27–36 |
|---|---|---|---|---|
| 1 | 469 | 7 | 96 | 366 |
| 2–4 | 618 | 145 | 256 | 217 |
| 5–6 | 113 | 23 | 62 | 28 |
| 7–8 | 82 | 26 | 31 | 25 |
| 9–12 | 121 | 49 | 37 | 35 |
| 13–16 | 94 | 27 | 53 | 14 |
| ≥ 17 | 584 | 479 | 105 | 0 |

r and d differ for about half the positions in bucket 2–4 and for 8–12% of the deeper buckets. r is often even (elimination by refutation), so the buckets include even r. The rarest cells (r 13–16 late in the game; r 5–6 early) need roughly 200–300k solved positions to reach about 500 per cell.

**Revision to 3.1.** A depth-4 alpha-beta engine in Python would be slower than the solver itself. Python sampling of safe-random play already runs at only about 80 positions/s per worker. Instead, generate **near-optimal games with the solver**: at every ply, play a random optimal move with probability 1 − ε, else a random safe move, and **keep every position along the game** with its scores. Each solve then produces a dataset position, at no extra cost. Mix: 50% near-optimal trajectories (ε ∈ {0.1, 0.3}), 50% safe-random. Split by game, not by position, since positions in one game are correlated. Budget: about 2.5M solved positions at about 100/s, roughly 7 hours.

### 3.6 Stage 1–2 results (2026-11-20)

Built on this machine in 5 h 50 min with 2 background workers. Over 2,059 guard checks, DNS failures were 2.1% (AdGuard baseline 2.9%), never 2 in a row, and free memory stayed at 58% or more.

| | |
|---|---|
| Solved positions | 2,507,251 from 87,776 games (eps 0.1 / 0.3 / 1.0 = 25 / 25 / 50% of games) |
| Train (`data/c4v2_train.npz`) | 1,683,332 after dedupe and removal of 40,424 test-equivalent positions |
| Test pool | 329,208 positions from 13,166 held-out games |
| Tactical set (`data/c4v2_tactical.npz`) | 12,991: 500 per (r bucket, stage) cell; 26 of 27 cells full (r ≥ 17 doesn't occur at plies 27–36) |
| General set (`data/c4v2_general.npz`) | 10,000, balanced by d |
| Overlap with train | 0 canonical positions, 0 games |
| r vs. ply correlation (tactical) | −0.23, down from −0.63 for d in the pilot test set |

**Baselines on the tactical set** (`results/c4v2-baselines/`):

| r | n | Chance | Win/block shortcut | Search k=1 | k=2 | k=4 | k=8 | k=12 | k=16 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1,491 | 0.20 | **1.00** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2 | 1,500 | 0.21 | **1.00** | 0.21 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 3–4 | 1,500 | 0.20 | 0.22 | 0.20 | 0.22 | 1.00 | 1.00 | 1.00 | 1.00 |
| 5–6 | 1,500 | 0.21 | 0.23 | 0.21 | 0.23 | 0.26 | 1.00 | 1.00 | 1.00 |
| 7–8 | 1,500 | 0.20 | 0.22 | 0.20 | 0.22 | 0.24 | 1.00 | 1.00 | 1.00 |
| 9–10 | 1,500 | 0.19 | 0.20 | 0.19 | 0.20 | 0.22 | 0.25 | 1.00 | 1.00 |
| 11–12 | 1,500 | 0.18 | 0.20 | 0.18 | 0.19 | 0.22 | 0.24 | 1.00 | 1.00 |
| 13–16 | 1,500 | 0.17 | 0.18 | 0.17 | 0.18 | 0.20 | 0.22 | 0.24 | 1.00 |
| ≥ 17 | 1,000 | 0.15 | 0.16 | 0.15 | 0.16 | 0.17 | 0.18 | 0.19 | 0.21 |

Validity checks 6.4 pass: search steps from chance to 1.0 exactly at k = r, and the shortcut solves only r ≤ 2, staying within 2 points of chance for r ≥ 3.

## 4. Models

All models: MLP mixer, d_model 256, about 1.25M parameters unless stated. Same input encoding and heads (policy over the optimal set, value).

| Model | Role | Thinking unit |
|---|---|---|
| **TRM-small** (main) | The hypothesis | One supervision step = T=1, n=1 = 4 layer applications. n_sup = 32, so *K* goes up to 32 steps, 128 layer applications |
| TRM-small, second seed | Is the result reproducible? | Same |
| ds-only (n=0) | Does latent recursion matter, or just repeated answer refinement? | Same step count |
| **Direct-L**, L ∈ {4, 8, 16, 32} | Non-recursive depth: is it recursion, or just more layers? | L layers, one pass, untied weights |
| TRM-wide (d_model 512), stretch | Can more iterations substitute for more parameters? | Same as TRM-small |

## 5. Training and baselines

### 5.1 Training

- **Objective:**
  - uniform deep supervision, which rewards answering as early as possible, so "steps needed" is measurable
  - halting head trained
  - **no** halting during training: every example runs all 32 steps, so every *K* is trained
- **Ablation:** linear step weighting (`--step-weight linear`) on TRM-small. It shows whether pressure to answer early changes *K\*(r)*.
- **Optimizer:**
  - AdamW, lr 2e-4, warmup 1,000 steps, **cosine decay** to 10% (to be added)
  - EMA 0.999, weight decay 0.1, batch 128, mirror augmentation
- **Length:** up to 150k steps. Stop early after 10 evaluations without +0.2 points on the tactical set's *K*=32 accuracy. `best.pt` keeps the best checkpoint.
- **Evaluations during training:** every 2,000 steps, on a 2,000-position slice of the tactical set, logging accuracy by step for each *r* bucket, so the depth signal is visible while the model trains.
- **Hardware:** this Mac mini, with `run_c4_train.sh` (guarded, resumable). At about 0.3 s per step, 150k steps is about 13 hours per TRM run. Direct models are faster.

### 5.2 Search baseline (no training)

Depth-limited alpha-beta with no evaluation function, which counts only proven wins and losses, at depths *k* = 1…20 on the same tactical set. By construction its accuracy-vs-depth curve for bucket *r* jumps at *k* = *r*. This is the **reference shape** for "search". It also confirms the *r* labels mean what they claim.

## 6. Evaluation and analysis

All final numbers on CPU (fp32). MPS changes about 6% of individual answers, so it's used only for development.

### 6.1 Primary analysis

For each model, for each *r* bucket, plot accuracy against *K* (iterations for TRM, layers for Direct-L). Then:

- **K\*(r)** = the smallest *K* at which accuracy reaches the midpoint between its *K*=1 value and its plateau. Bootstrap over positions, 2,000 resamples, for 95% confidence intervals.
- **Slope** of *K\*(r)* against *r*, by weighted least squares, with a bootstrap CI.
- **Per-position steps needed:** the first *K* after which the answer stays correct. Rank correlation with *r* is reported *within each game stage* (partial correlation controlling for ply and number of legal moves).

### 6.2 Success criteria (fixed now)

The hypothesis is **supported** if all of the following hold for TRM-small, in both seeds:

1. The slope of *K\*(r)* against *r* is positive, with a 95% CI that excludes 0.
2. The within-stage rank correlation between steps needed and *r* is at least 0.3. For comparison, the Sudoku sanity check got 0.29 with a cruder difficulty measure; the Connect Four pilots got 0.05–0.09.
3. For *r* ≥ 7, accuracy at *K*=32 exceeds accuracy at *K*=1 by at least 10 points. Otherwise the iterations aren't doing the work.

It's **partially supported** if 1 holds but 2 or 3 doesn't. It's **not supported** if 1 fails.

### 6.3 Secondary analyses

- **Recursion vs. depth:** do Direct-L models show the same *r* structure as L grows? If they do, "sequential depth" helps, whether tied or not. If only TRM shows it, weight-tying and iteration matter.
- **Latent recursion vs. answer refinement:** TRM vs. ds-only.
- **Halting:** does the learned halting step correlate with *r*, within each game stage?
- **Extrapolation:** run TRM-small for *K* = 48 and 64, beyond training. Do the deep buckets keep improving?
- **Parameters vs. iterations (stretch):** compare TRM-wide at small *K* with TRM-small at large *K*.
- **Continuity:** all models on the general set and Pons' benchmark.

### 6.4 Validity checks before reading results

- **Search baseline (5.2):** its curves step at *k* = *r*. If not, the *r* labels are wrong.
- **Chance and shortcut baselines per *r*:** random legal move; "win now, else block, else random". If the shortcut solves the high-*r* buckets, the filter needs tightening.
- **Leakage:** no test position or its mirror image appears in training (assert in code).
- **Label shuffle control:** rerun the analysis with *r* shuffled within game stage. The slope should vanish.

## 7. Stages and compute (this machine)

| Stage | What | Est. time | Safety |
|---|---|---|---|
| 0 | Solver probe (3.4), *r*-label code + tests, search baseline code | 1 day of coding, 10-min probe | Guarded, background priority |
| 1 | Build dataset (here if Stage 0 allows, else elsewhere); validity checks 6.4 | Hours (depends on Stage 0) | Guarded, sharded, resumable |
| 2 | Search baseline and shortcut baselines on the tactical set | Minutes | Guarded |
| 3 | Train TRM-small seed 0; check the during-training *r* curves at about 30k steps (go/no-go) | About 13 h | `run_c4_train.sh` |
| 4 | TRM-small seed 1, ds-only, Direct-4/8/16/32, linear-weight ablation | About 3–4 days total | One at a time |
| 5 | Final CPU evaluations, analysis, plots, write-up | About 1 day | Guarded |

**Stage 3 notes (2026-11-21).** With 32 supervision steps and no halting, each position stays in the batch for 32 consecutive updates, so only 4 new positions enter per update (batch 128). Training accuracy at the final step therefore reads about 100% (the position has been trained on 31 times). That isn't a measure of generalization. At step 8,000 (32k distinct positions), held-out accuracy at K=32 was 49% on train positions, 52% on the general set and 31% on the tactical set. Chance is about 38%, 38% and 19%. Tactical accuracy will lag general accuracy for a long time, so early stopping on the tactical slice would stop the run too soon. It is turned off (`PATIENCE=0`), and the cosine schedule runs to 150k steps (600k positions, about 10 h).

**Revised go/no-go: at 60k steps (240k positions), not 30k.** r = 1 and r = 2 (win now, must block) must be clearly learned, above 60% at K = 32, before deeper buckets can mean anything. If they aren't, per-example learning is too slow for this setup (revisit n_sup 16, d_model 512 or batch size) before running the baselines.

**Diagnostics during Stage 3 (2026-11-21, 01:00–02:30).** Tactical accuracy sat at 30–33% and win-in-one at about 40% from 32k to 104k positions seen, so I checked for a bug or bad data:
- Data checks: tokens match move strings, labels match solver scores, every r=1 position's labelled move wins at once.
- The pilot small-step model (v1 data, 550k positions) scores 70% on the v2 tactical set: 98% on r=1, 94% on r=2, then 81% (r 3–4) falling to about 55% from r=5 on. So the set is learnable (`results/c4v2-pilot-model/`).
- Same config (T=1, n=2, n_sup 16), 4,000 steps, trained on v1 vs. v2 data: tactical 33.7% vs. 36.6%, r=1 54% vs. 40%, r=2 36% vs. 47%. So the data isn't the problem.
- Conclusion: no bug. These tactics are learned slowly early in training for every setup. The main run resumed unchanged, and the 60k-step go/no-go stands.

**Go/no-go at 60k steps (2026-11-21 04:37): NO-GO.** After 240k positions, at K=32 on the tactical slice: r=1 40%, r=2 50% (the bar was 60%); overall 36.6% (best 37.4%). From about 40k steps on, step 1 often beats step 32 on r=1 (for example 49% vs. 42% at 46k), so later steps tend to undo correct early answers. The run is stopped (resumable at step 60,074); logs are in `results/c4v2-trm-small-s0/`. DNS over the run: 2.0% failed probes (1,353 checks), never 2 in a row; free memory ≥ 48%.

**Original go/no-go after Stage 3 at 30k steps:** if accuracy by step on the tactical set is flat in *K* for every *r* bucket, stop before spending days on baselines. Revisit step size (smaller steps) or the data first.

## 8. Code changes needed

1. `connect4.py`:
   - store per-column scores
   - *r* label and its tests
   - engine and near-optimal sampling
   - canonical-form dedupe and split
   - tactical-set builder
2. Guarded, sharded, resumable data-build runner (modelled on `run_c4_train.sh`).
3. Search baseline: depth-limited alpha-beta. Use C++ or bitboards for speed, or reuse the solver's `Position` code with a depth cap.
4. `c4-train`:
   - cosine decay
   - evaluation on the tactical slice by *r*
   - `--n-sup 32`
5. `c4-eval`: *r* buckets, game-stage strata, bootstrap CIs, *K\*(r)*, and the shuffle control.
6. CPU evaluation path for final numbers.
