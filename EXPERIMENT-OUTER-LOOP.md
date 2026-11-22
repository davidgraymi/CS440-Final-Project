# Experiment plan v3: put the thinking in the outer loop

Status: proposed 2026-11-21, after the Stage 3 no-go in [EXPERIMENT.md](EXPERIMENT.md). This plan keeps that document's dataset, labels, evaluation sets and safety setup. It changes the model, the training objective and the main measurement.

## 1. What we learned, and the claim we now test

All five Connect Four models score the same at step 1 and at step 32 on the hard tactical set (`results/c4v2-model-comparison/`). Looping does help *within* a step: 42 passes of a 2-layer net beat 8 different layers, 80.5% to 70.7%. But no model gets better with more outer steps, and the learned stop vote stops at step 1 on nearly every position.

We attribute this to three properties of the current training. Each one is a design choice we can change:

| Cause | Current setting | Effect |
|---|---|---|
| Each step is very deep | T=3, n=6: 42 layer passes | One step solves what the model can solve, so later steps have nothing to add |
| No credit across steps | `detach` between steps; every step graded on its own | A step is never rewarded for setting up a later step (like RL with discount 0) |
| All-or-nothing target | Best move only | Half-finished lookahead earns nothing, because the graded move is still wrong |

**Claim tested:** when each outer step is shallow, credit flows across steps, and the target rewards partial progress, the model learns to take **more outer steps on positions that need deeper lookahead**. Here r is the required search depth from EXPERIMENT.md 3.2.

## 2. Model

- TRM with **T=1** (one round per step) and **n ∈ {1, 2}**: 4 or 6 layer passes per step. 2 layers, d_model 256, MLP mixer, as before.
- Up to **K_max = 24 outer steps**. At most 144 layer passes, about 3.4× the overnight model's 42 per step.
- Heads read the answer state y after every step:
  - **Per-column head (new):** for each of the 7 columns, the outcome of playing there, as 9 classes ordered worst to best: loss in 1–2 / 3–6 / 7–14 / 15+ plies, draw, win in 15+ / 7–14 / 3–6 / 1–2 plies. Full columns are masked. The targets come from the `scores` already stored in `data/c4v2_*.npz`. On every tactical position the optimal column has the single best class.
  - **Move:** the existing move head (argmax over columns), so move accuracy stays comparable with every earlier model. The per-column head is an extra training target, and its accuracy by step is reported.
  - **Stop head:** the probability λ_k of stopping at step k, read from the first slot as today.

## 3. Training objective

**Unrolled episodes replace the rolling batch.** Each update takes 128 new positions and runs all K_max steps on them. The forced-32 run let only 4 new positions into each update, which is why it learned so slowly.

**Halting-weighted loss (PonderNet-style).** The chance of stopping exactly at step k is p_k = λ_k ∏_{j<k}(1−λ_j). The leftover probability is assigned to K_max. The loss is:

  L = Σ_k p_k · L_task(k) + β · KL(p ‖ Geometric(prior))

L_task(k) is the per-column loss at step k, averaged over legal columns, plus 0.5 × the move loss.

- Steps the model is likely to stop at count most. So the model is graded mostly on the answer it would actually give.
- The geometric prior acts as a cost per step. Its mean is the prior expected number of steps, set to 6. β ∈ {0.01, 0.1}.

**Credit across steps: window W.** The carried state is detached every W steps. W=1 means no credit across steps, as in the old training. W=0 means never detached: full credit through all K_max steps.

**Memory: gradient checkpointing per step.** During training only the state between steps is kept. Each step's internals are recomputed when the learning signal passes back through it. Measured on this machine (batch 128, n=1): about 13 MB per step instead of 870 MB. Peak is about 2.2 GB for K_max=24 whatever W is, under the 0.3 MPS cap. So window length is chosen for learning quality, not memory.

**Speed: measured at 21 new positions/s** (6.2 s per update of 128 positions × 24 steps). That's the same rate as forced-32, because every position now gets up to 24 steps during training. To recover time once the model learns to stop, an update stops unrolling when every position in the batch has stopped with probability > 99% (`--stop-eps 0.01`). Evaluation always runs all K_max steps.

**Speed-ups (measured 2026-11-21 with `neural-cost` 0.12.0 and full-update benchmarks, batch 128, 24 steps, M1 8-core GPU):**
- **Roofline analysis.** One forward step takes 51.8 ms (median of 20 runs), with 100% of its 59 traced operations modelled. That's 56% of the roofline bound: memory-bound, 670 GFLOP/s. The gap breaks down into unfused elementwise traffic (15%), serialization (12%), peak derate (10%), dispatch (1%) and unexplained (19%).
- **Full training update:**

  | Setting | Time per update | Speed |
  |---|---|---|
  | fp32 | 6.2 s | 20.6 positions/s |
  | `--amp fp16` | 5.2 s | 24.3 positions/s |
  | `--compile` | 4.7 s | 27.3 positions/s |
  | **`--compile --amp fp16`** | **4.0 s** | **32.1 positions/s (1.57×)** |

  bf16 is slower (19.4/s): the M1 has no bf16 hardware. Batch 256 gives no gain (19.4/s at fp32), because the GPU is already saturated.
- **Accuracy check.** Losses over 6 updates match fp32 to within 0.0005. In a 100-update run on real data, the fp16 gradient scale stayed at 65536 with no skipped updates.
- **Pilots and the main run use `--compile --amp fp16`.**

**Health check:** every 100 updates the log records the size of the learning signal reaching the state at steps 1, K_max/2 and K_max. In the smoke test it was 1.7e-2, 4.5e-3 and 3.1e-3: it reaches step 1 without fading or blowing up.

Everything else stays as in EXPERIMENT.md 5.1: AdamW, lr 2e-4, cosine schedule, EMA 0.999, horizontal-mirror augmentation, guard and resume via `scripts/run_c4_train.sh`.

## 4. Pilot: a 2 × 2 grid to find what matters (about 10 h total)

Each cell trains for 2.5 h (about 190k positions) on v2 data, n=1, K_max=24, β=0.1, prior mean 6:

| | Move target only (`--col-weight 0`) | Per-column target (`--col-weight 1`) |
|---|---|---|
| **W = 1** (no credit across steps) | A: control | B |
| **W = 0** (full credit) | C | D: full design |

W=8 was dropped: memory no longer forces a short window, and the pilot is better spent on the two ends.

**Go to the full run if the best cell (expected D) meets all three:**
1. For r ≥ 3, accuracy rises with K, from K=1 up to the step it stops at, by at least 5 points. These are the curves that were flat in every model so far.
2. The stopping step differs across r buckets (Kruskal–Wallis p < 0.01) and rises with r: mean within-stage Spearman ≥ 0.1. The effect-size bar was added after a 30-update dry run, in which untrained models reached p < 0.01 at Spearman 0.04.
3. r = 1 and r = 2 accuracy at the stop step ≥ 60%, so it is actually learning the game.

190k positions is less than any earlier model saw: the overnight TRM saw about 2.2M. So the pilot judges the shape of the curves, not final accuracy. If no cell passes, stop and write up the negative result. If only B or C passes, that tells us which of the two changes did the work, and the full run uses that cell.

## 5. Full run and controls (about 3 days)

- **Main:** the winning pilot cell, 2 seeds, about 12 h each.
- **Fixed-steps control:** same model, stop head off, always runs K steps, K ∈ {1, 4, 8, 24}. Shows how much accuracy is bought by steps without any adaptive stopping.
- **One-pass baseline:** already have it (direct-overnight, 70.7% on the tactical set). Add a one-pass model of matched depth (about the mean passes used by Main), to separate "more compute" from "chosen compute".

## 6. Measurements and success criteria (fixed before running)

Evaluated on the full tactical set (12,991 positions), with 95% bootstrap CIs over positions:

1. **Primary: stopping step tracks lookahead.** Spearman correlation between the stopping step and r, computed within each game stage (plies 6–16, 17–26, 27–36) so position count isn't the driver. Success: ≥ 0.3 in every stage, with the CI excluding 0.
2. **Extra steps buy accuracy where they are taken.** For r ≥ 7, accuracy at the stopping step minus accuracy at K=1 is ≥ 10 points (the same bar as EXPERIMENT.md 6.2, criterion 3).
3. **Adaptive beats fixed at equal cost.** Main's accuracy is ≥ the fixed-steps control's at the same mean number of steps.
4. **Shuffle control.** With r shuffled within stage, criterion 1's correlation falls to about 0.

Also reported: accuracy vs. K by r bucket (as in the charts so far), the stopping-step distribution by r, and per-column accuracy vs. K, which shows partial progress.

**Not used as a target:** grading step k against a k-move search. That would build the result into the training.

## 7. Code changes

Done (2026-11-21):
1. `models.py`: optional per-column head (`head="game_cols"`); old checkpoints are unaffected.
2. `c4_ponder.py` (new; `c4-ponder`, `c4-ponder-eval`):
   - per-column targets
   - unrolled episodes with window W and per-step gradient checkpointing
   - halting-weighted loss with geometric KL
   - early stop of the unroll
   - eval of accuracy by step, stopping step, per-column accuracy, by r, and stop-vs-r Spearman by stage with bootstrap CI and the shuffle control
   - flags: `--window`, `--col-weight`, `--beta`, `--prior-steps`, `--k-max`, `--stop-eps`
3. `scripts/run_c4_train.sh`: `TRAINER=ponder` runs the new trainer under the same guard.
4. `tests/test_ponder.py`:
   - per-column targets match a slow reference and the solver's optimal move
   - mirror augmentation flips every target
   - halting and prior are distributions
   - W=1 blocks credit to earlier steps and W=0 passes it
   - checkpointing gives identical gradients
   - early unroll stop
   - end-to-end train and eval smoke

## 8. Timeline on this machine (guarded as before)

| Stage | What | Time |
|---|---|---|
| 1 | Code + tests + memory and speed check | Done 2026-11-21 |
| 2 | 2 × 2 pilot | About 10 h |
| 3 | Main (2 seeds) + controls | About 3 days |
| 4 | Analysis, charts, write-up | 1 day |
