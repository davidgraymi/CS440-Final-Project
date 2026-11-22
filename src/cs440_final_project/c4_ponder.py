"""Outer-loop thinking on Connect Four (EXPERIMENT-OUTER-LOOP.md).

    c4-ponder --run-name c4p-D --window 0 --col-weight 1      # full design
    c4-ponder-eval --ckpt runs/c4p-D/ckpt.pt                  # stopping step vs. required depth

Differences from c4_train.py:
  * Each update runs whole episodes: 128 new positions, every one unrolled for
    K_max outer steps (no rolling batch).
  * Shallow steps (default T=1, n=1: 4 layer passes per step).
  * Halting-weighted loss (PonderNet): the stop head gives lambda_k, the chance
    of stopping at step k given it got that far. p_k = lambda_k * prod_{j<k}(1 - lambda_j),
    and whatever is left over is assigned to K_max. Loss:
        sum_k p_k * L_task(k)  +  beta * KL(p || truncated geometric with mean ~ prior_steps)
  * Credit across steps: the carried state (y, z) is detached every `window` steps
    (window 1: no credit across steps, as in the old training; 0: never detached).
    Each step is gradient-checkpointed, so memory stays about one step's worth
    whatever the window.
  * Optional per-column target (--col-weight): for every legal column, the outcome
    of playing it (win / draw / loss and how fast), from the solver scores. It gives
    partial credit: a step can get the easy columns right before the hard ones.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset

from .c4_train import ROOT, RUNS_DIR, load, worker_init
from .connect4 import INVALID, VOCAB_SIZE
from .models import N_COL_CLASSES, ModelConfig, build_model
from .train import ExampleStream, pick_device

# Plies-to-end buckets for a column's outcome: 1-2, 3-6, 7-14, 15+.
DIST_EDGES = (2, 6, 14)
DRAW_CLASS = 4
IGNORE = -100  # full column
STAGES = ((6, 16), (17, 26), (27, 36))


# ---------------------------------------------------------------------- targets


def col_classes(scores: np.ndarray, n_moves: np.ndarray) -> np.ndarray:
    """(N, 7) int16 outcome class of playing each column, ordered worst to best:
    0..3 loss (fastest loss first), 4 draw, 5..8 win (fastest win last); IGNORE for full columns.
    Distance = plies to the end (connect4.depth_from_score, vectorized)."""
    s = scores.astype(np.int64)
    n = n_moves.astype(np.int64)[:, None]
    dist = np.where(s > 0, 2 * ((43 - n) // 2 - s) + 1, 2 * ((42 - n) // 2 + s) + 2)
    b = np.searchsorted(DIST_EDGES, dist)  # 0 = fastest
    out = np.where(s > 0, 8 - b, b)
    out = np.where(s == 0, DRAW_CLASS, out)
    return np.where(s == INVALID, IGNORE, out).astype(np.int16)


def pack(d: dict) -> np.ndarray:
    """(N, 22) int16: optimal (7), legal (7), value class (1), column classes (7)."""
    if "cols" not in d:
        d["cols"] = col_classes(d["scores"], d["n_moves"])
    return np.concatenate([d["optimal"], d["legal"], (d["value"] + 1)[:, None], d["cols"]], 1).astype(np.int16)


class PonderDataset(Dataset):
    def __init__(self, d: dict, mirror: bool):
        self.x = np.ascontiguousarray(d["tokens"], dtype=np.uint8)
        self.y = pack(d)
        self.mirror = mirror
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x, y = self.x[i], self.y[i]
        if self.mirror and self.rng.random() < 0.5:
            x = x.reshape(6, 7)[:, ::-1].reshape(-1)
            y = np.concatenate([y[6::-1], y[13:6:-1], y[14:15], y[21:14:-1]])
        return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))


# ---------------------------------------------------------------------- model


def one_step(model, x, y, z):
    for _ in range(model.cfg.T):
        y, z = model.latent_recursion(x, y, z)
    return y, z


def episode(model, tokens, k_max: int, window: int = 0, ckpt: bool = True, hook=None, stop_eps: float = 0.0,
            step_fn=None):
    """Unroll up to k_max outer steps. Returns per-step lists: policy (B,7), value (B,3), cols (B,7,C), q (B,).
    stop_eps > 0: stop unrolling once every position has stopped with probability > 1 - stop_eps
    (the leftover chance is then assigned to the last step run; see halting_dist)."""
    step_fn = step_fn or one_step
    x = model.embed_inputs(tokens)
    y, z = model.init_carry(len(tokens), tokens.device)
    outs = dict(policy=[], value=[], cols=[], q=[])
    survive = torch.ones(len(tokens), device=tokens.device)
    for k in range(k_max):
        if stop_eps and k and survive.max().item() < stop_eps:
            break
        if k and window and k % window == 0:
            y, z = y.detach(), z.detach()
        if ckpt and torch.is_grad_enabled():
            y, z = checkpoint(step_fn, model, x, y, z, use_reentrant=False)
        else:
            y, z = step_fn(model, x, y, z)
        if hook is not None and y.requires_grad:
            hook(k, y)
        o = model.answer_logits(y, tokens)
        outs["policy"].append(o.policy)
        outs["value"].append(o.value)
        outs["cols"].append(model.column_logits(y) if hasattr(model, "col_head") else None)
        outs["q"].append(model.q_head(y[:, 0]).float().squeeze(-1))
        if stop_eps:
            survive = survive * (1 - torch.sigmoid(outs["q"][-1].detach()))
    return outs


def halting_dist(q: torch.Tensor) -> torch.Tensor:
    """q (B, K) stop logits -> p (B, K): chance of stopping exactly at each step; the rest goes to step K."""
    lam = torch.sigmoid(q)
    lam = torch.cat([lam[:, :-1], torch.ones_like(lam[:, -1:])], 1)
    survive = torch.cumprod(torch.cat([torch.ones_like(lam[:, :1]), 1 - lam[:, :-1]], 1), 1)
    return lam * survive


def geometric_prior(k_max: int, prior_steps: float, device) -> torch.Tensor:
    lp = 1.0 / prior_steps
    k = torch.arange(k_max, device=device, dtype=torch.float32)
    p = lp * (1 - lp) ** k
    p[-1] = (1 - lp) ** (k_max - 1)
    return p


def stop_step(p: torch.Tensor) -> torch.Tensor:
    """Deterministic stopping step (1-based): the first step where the chance of having stopped reaches 1/2."""
    return (torch.cumsum(p, 1) >= 0.5).float().argmax(1) + 1


# ---------------------------------------------------------------------- losses


def step_task_loss(policy, value, cols, y, value_weight: float, col_weight: float):
    """Per-example task loss at one step, and whether the argmax move is optimal."""
    optimal = y[:, :7].bool()
    logp = F.log_softmax(policy, -1)
    loss = -torch.logsumexp(logp.masked_fill(~optimal, -1e9), -1)
    loss = loss + value_weight * F.cross_entropy(value, y[:, 14], reduction="none")
    if col_weight and cols is not None:
        tgt = y[:, 15:22]
        ce = F.cross_entropy(cols.reshape(-1, N_COL_CLASSES), tgt.reshape(-1), reduction="none",
                             ignore_index=IGNORE).view_as(tgt)
        legal = (tgt != IGNORE).float()
        loss = loss + col_weight * (ce * legal).sum(1) / legal.sum(1).clamp(min=1)
    ok = optimal.gather(1, policy.argmax(-1, keepdim=True)).squeeze(1)
    return loss, ok


def ponder_loss(outs, y, args):
    K = len(outs["q"])  # may be < k_max when the unroll stopped early
    prior = geometric_prior(K, args.prior_steps, y.device)
    task, ok = zip(*(step_task_loss(outs["policy"][k], outs["value"][k], outs["cols"][k], y,
                                    args.value_weight, args.col_weight) for k in range(K)))
    task, ok = torch.stack(task, 1), torch.stack(ok, 1)  # (B, K)
    if getattr(args, "fixed_steps", 0):  # control: always K steps, graded at the last one, no stop vote
        p = torch.zeros_like(task)
        p[:, -1] = 1
        rec, kl = task[:, -1], torch.zeros_like(task[:, -1])
    else:
        p = halting_dist(torch.stack(outs["q"], 1))
        rec = (p * task).sum(1)
        kl = (p * (torch.log(p.clamp(min=1e-10)) - torch.log(prior))).sum(1)
    loss = (rec + args.beta * kl).mean()
    with torch.no_grad():
        steps = torch.arange(1, K + 1, device=p.device, dtype=p.dtype)
        stats = dict(loss=loss.item(), task=rec.mean().item(), kl=kl.mean().item(),
                     exp_steps=(p * steps).sum(1).mean().item(), unrolled=K,
                     acc_k1=ok[:, 0].float().mean().item(), acc_kmax=ok[:, -1].float().mean().item(),
                     acc_at_stop=ok.gather(1, stop_step(p)[:, None] - 1).float().mean().item())
    return loss, stats


# ---------------------------------------------------------------------- evaluation


@torch.no_grad()
def run_eval(model, d: dict, device, k_max: int, batch_size: int = 256, fixed_steps: int = 0) -> dict:
    """Per position and step: move optimal, fraction of legal columns classified right, and p (stop distribution)."""
    was = model.training
    model.eval()
    y_all = torch.from_numpy(pack(d).astype(np.int64))
    x_all = torch.from_numpy(d["tokens"].astype(np.int64))
    N = len(x_all)
    move_ok = np.zeros((N, k_max), bool)
    col_acc = np.full((N, k_max), np.nan, np.float32)
    p_all = np.zeros((N, k_max), np.float32)
    for i in range(0, N, batch_size):
        x, y = x_all[i:i + batch_size].to(device), y_all[i:i + batch_size].to(device)
        outs = episode(model, x, k_max, ckpt=False)
        optimal = y[:, :7].bool()
        for k in range(k_max):
            move_ok[i:i + len(x), k] = optimal.gather(1, outs["policy"][k].argmax(-1, keepdim=True)).squeeze(1).cpu().numpy()
            if outs["cols"][k] is not None:
                tgt = y[:, 15:22]
                legal = tgt != IGNORE
                right = (outs["cols"][k].argmax(-1) == tgt) & legal
                col_acc[i:i + len(x), k] = (right.sum(1).float() / legal.sum(1).clamp(min=1)).cpu().numpy()
        if fixed_steps:
            p_all[i:i + len(x), fixed_steps - 1] = 1
        else:
            p_all[i:i + len(x)] = halting_dist(torch.stack(outs["q"], 1)).cpu().numpy()
    model.train(was)
    return dict(move_ok=move_ok, col_acc=col_acc, p=p_all)


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def summarize(res: dict, d: dict, n_boot: int = 0, seed: int = 0) -> dict:
    """Accuracy at the stopping step and by step, overall and by required depth r; stop step vs. r."""
    from .c4data import R_BUCKETS

    move_ok, p = res["move_ok"], res["p"]
    K = move_ok.shape[1]
    stop = (np.cumsum(p, 1) >= 0.5).argmax(1) + 1
    exp_steps = (p * np.arange(1, K + 1)).sum(1)
    at_stop = move_ok[np.arange(len(stop)), stop - 1]
    out = dict(n=len(stop), acc_at_stop=round(float(at_stop.mean()), 4),
               acc_by_step=[round(float(v), 4) for v in move_ok.mean(0)],
               mean_stop=round(float(stop.mean()), 3), mean_exp_steps=round(float(exp_steps.mean()), 3))
    if not np.isnan(res["col_acc"]).all():
        out["col_acc_by_step"] = [round(float(v), 4) for v in np.nanmean(res["col_acc"], 0)]
    if "req_depth" not in d:
        return out
    r, ply = d["req_depth"], d["n_moves"]
    out["by_r"] = {}
    for lo, hi in R_BUCKETS:
        m = (r >= lo) & (r <= hi)
        if m.any():
            out["by_r"][f"{lo}-{hi}"] = dict(
                n=int(m.sum()), acc_at_stop=round(float(at_stop[m].mean()), 4),
                mean_stop=round(float(stop[m].mean()), 3), mean_exp_steps=round(float(exp_steps[m].mean()), 3),
                acc_by_step=[round(float(v), 4) for v in move_ok[m].mean(0)])
    rng = np.random.default_rng(seed)
    out["spearman_stop_vs_r_by_stage"] = {}
    for lo, hi in STAGES:
        m = (ply >= lo) & (ply <= hi) & (r >= 1)
        if m.sum() < 3:
            continue
        rho = spearman(exp_steps[m], r[m])
        row = dict(n=int(m.sum()), rho_exp_steps=round(rho, 4), rho_stop=round(spearman(stop[m], r[m]), 4))
        if n_boot:
            idx = np.flatnonzero(m)
            bs = [spearman(exp_steps[s], r[s]) for s in (rng.choice(idx, len(idx)) for _ in range(n_boot))]
            row["ci95"] = [round(float(np.nanpercentile(bs, 2.5)), 4), round(float(np.nanpercentile(bs, 97.5)), 4)]
            shuf = r[idx].copy()
            rng.shuffle(shuf)
            row["rho_shuffled_r"] = round(spearman(exp_steps[idx], shuf), 4)
        out["spearman_stop_vs_r_by_stage"][f"{lo}-{hi}"] = row
    return out


# ---------------------------------------------------------------------- training


def train_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True)
    p.add_argument("--data-prefix", default=str(ROOT / "data" / "c4v2"))
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--T", type=int, default=1)
    p.add_argument("--k-max", type=int, default=24, help="outer steps unrolled per episode")
    p.add_argument("--window", type=int, default=0,
                   help="detach the carried state every this many steps (1: no credit across steps; 0: never)")
    p.add_argument("--col-weight", type=float, default=1.0, help="weight of the per-column target (0: move only)")
    p.add_argument("--value-weight", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.1, help="weight of KL(stop distribution || geometric prior)")
    p.add_argument("--prior-steps", type=float, default=6.0, help="mean of the geometric prior on steps")
    p.add_argument("--fixed-steps", type=int, default=0,
                   help="control: no stop vote; always run this many steps (sets k-max) and grade the last one")
    p.add_argument("--stop-eps", type=float, default=0.01,
                   help="training: stop unrolling once every position has stopped with prob > 1 - this (0: always k_max)")
    p.add_argument("--compile", action="store_true", help="torch.compile one outer step (1.33x on the M1 GPU)")
    p.add_argument("--amp", choices=["none", "fp16"], default="none",
                   help="fp16 autocast with gradient scaling for training (1.18x; 1.57x with --compile on the M1 GPU)")
    p.add_argument("--no-ckpt", dest="ckpt", action="store_false", help="no gradient checkpointing (more memory)")
    p.add_argument("--steps", type=int, default=20_000, help="optimizer updates (each = batch-size new positions)")
    p.add_argument("--time-limit-min", type=float, default=0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default="cosine")
    p.add_argument("--min-lr-frac", type=float, default=0.1)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--no-mirror", dest="mirror", action="store_false")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-split", default="tactical")
    p.add_argument("--eval-limit", type=int, default=2000)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)


def build(args, device):
    cfg = ModelConfig(vocab_size=VOCAB_SIZE, seq_len=42, d_model=args.d_model, n_heads=8, mixer="mlp",
                      n_prefix=1, head="game_cols", n_layers=args.n_layers, n=args.n, T=args.T, n_sup=args.k_max)
    model = build_model("trm", cfg).to(device)
    with torch.no_grad():  # start the stop head at the prior's per-step stop chance
        lp = 1.0 / args.prior_steps
        model.q_head.bias.fill_(math.log(lp / (1 - lp)))
    return cfg, model


def lr_factor(args):
    def f(s):
        warm = min(1.0, (s + 1) / max(1, args.warmup))
        if args.lr_schedule == "constant":
            return warm
        frac = min(1.0, max(0.0, (s - args.warmup) / max(1, args.steps - args.warmup)))
        return warm * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * frac)))
    return f


def train_main(argv=None):
    args = train_args(argv)
    if args.fixed_steps:
        args.k_max, args.stop_eps = args.fixed_steps, 0.0
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    cfg, model = build(args, device)
    step_fn = torch.compile(one_step) if args.compile else one_step
    amp = args.amp == "fp16"
    autocast = torch.autocast(device.type, dtype=torch.float16, enabled=amp)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True)

    train = load(f"{args.data_prefix}_train.npz")
    if args.train_limit:
        train = {k: v[:args.train_limit] for k, v in train.items()}
    test = load(f"{args.data_prefix}_{args.eval_split}.npz")
    idx = np.sort(np.random.default_rng(0).permutation(len(test["tokens"]))[:args.eval_limit])
    test = {k: v[idx] for k, v in test.items()}
    t0 = time.time()
    loader = DataLoader(PonderDataset(train, args.mirror), args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.workers, worker_init_fn=worker_init, persistent_workers=args.workers > 0)
    print(f"labels ready in {time.time() - t0:.0f}s", flush=True)
    stream = ExampleStream(loader, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor(args))

    run_dir = RUNS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for p in model.parameters())
    (run_dir / "config.json").write_text(json.dumps(dict(args=vars(args), model=asdict(cfg), kind="trm",
                                                         params=n_params, layer_passes_per_step=model.effective_depth(),
                                                         device=str(device)), indent=2))
    print(f"{args.run_name}: {n_params / 1e6:.2f}M params, {model.effective_depth()} layer passes/step, "
          f"k_max {args.k_max}, window {args.window}, col_weight {args.col_weight}, beta {args.beta}, "
          f"train {len(train['tokens'])}, device {device}", flush=True)

    start, best = 0, -1.0
    ckpt_path = run_dir / "ckpt.pt"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"])
        ema.module.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        ema.n_averaged.fill_(ck["n_averaged"])
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        start, best = ck["step"], ck.get("best", -1.0)
        torch.manual_seed(args.seed + start)
        print(f"resumed at update {start}", flush=True)

    def save(step, path=ckpt_path):
        tmp = path.with_suffix(".tmp")
        torch.save(dict(model=model.state_dict(), ema=ema.module.state_dict(), cfg=asdict(cfg), kind="trm",
                        step=step, opt=opt.state_dict(), sched=sched.state_dict(),
                        n_averaged=int(ema.n_averaged.item()), best=best, args=vars(args),
                        scaler=scaler.state_dict()), tmp)
        tmp.replace(path)

    stop_signal = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda s, f: stop_signal.append(s))

    # learning-signal size reaching the answer state at a few steps (does credit survive long windows?)
    probe = sorted({0, args.k_max // 2, args.k_max - 1})
    grad_norms = {}

    def hook(k, y):
        if k in probe:
            y.register_hook(lambda g, k=k: grad_norms.__setitem__(k, g.norm().item()))

    with open(run_dir / "log.jsonl", "a") as log:
        def emit(rec):
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()

        t0, window, step = time.time(), [], start
        for step in range(start + 1, args.steps + 1):
            if stop_signal:
                save(step - 1)
                emit(dict(step=step - 1, stopped=f"signal {stop_signal[0]}", resumable=True))
                return
            x, y = stream.take(args.batch_size)
            with autocast:
                outs = episode(model, x, args.k_max, args.window, args.ckpt, hook if step % 100 == 0 else None,
                               stop_eps=args.stop_eps, step_fn=step_fn)
                loss, stats = ponder_loss(outs, y, args)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            stats["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip or 1e9))
            scaler.step(opt)  # skips the update if fp16 gradients overflowed
            scaler.update()
            sched.step()
            ema.update_parameters(model)
            window.append(stats)
            if step % 100 == 0:
                rec = dict(step=step, sec=round(time.time() - t0, 1),
                           **{k: round(float(np.mean([w[k] for w in window])), 4) for k in window[0]})
                unscale = 1.0 / scaler.get_scale() if amp else 1.0  # hooks see fp16-scaled gradients
                rec["state_grad_norm"] = {str(k + 1): f"{v * unscale:.2e}" for k, v in sorted(grad_norms.items())}
                if amp:
                    rec["grad_scale"] = scaler.get_scale()
                if device.type == "mps":
                    rec["mps_mb"] = round(torch.mps.driver_allocated_memory() / 2**20)
                emit(rec)
                window = []
            out_of_time = args.time_limit_min > 0 and time.time() - t0 > 60 * args.time_limit_min
            if step % args.eval_every == 0 or step == args.steps or out_of_time:
                res = run_eval(ema.module, test, device, args.k_max, args.eval_batch_size, args.fixed_steps)
                if device.type == "mps":
                    torch.mps.empty_cache()
                summ = summarize(res, test)
                best = max(best, summ["acc_at_stop"])
                emit(dict(step=step, eval=summ))
                save(step)
            if out_of_time:
                emit(dict(step=step, stopped="time limit", minutes=round((time.time() - t0) / 60, 1)))
                break
        (run_dir / "done.json").write_text(json.dumps(dict(step=step, minutes=round((time.time() - t0) / 60, 1))))


def eval_main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a c4-ponder checkpoint on a full split.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-prefix", default=str(ROOT / "data" / "c4v2"))
    p.add_argument("--split", default="tactical")
    p.add_argument("--k-max", type=int, default=None, help="default: trained k_max")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    device = pick_device(args.device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model = build_model("trm", ModelConfig(**ck["cfg"]))
    model.load_state_dict(ck["ema"])
    model.to(device)
    d = load(f"{args.data_prefix}_{args.split}.npz")
    k_max = args.k_max or ck["cfg"]["n_sup"]
    fixed = ck.get("args", {}).get("fixed_steps", 0)
    res = run_eval(model, d, device, k_max, args.batch_size, fixed)
    summ = summarize(res, d, n_boot=args.bootstrap)
    out = Path(args.out) if args.out else Path(args.ckpt).parent / f"eval-{args.split}-k{k_max}.json"
    out.write_text(json.dumps(dict(ckpt=args.ckpt, step=ck["step"], **summ), indent=2))
    np.savez_compressed(out.with_suffix(".npz"), **res, n_moves=d["n_moves"], **({"req_depth": d["req_depth"]} if "req_depth" in d else {}))
    print(json.dumps({k: v for k, v in summ.items() if k != "by_r"}, indent=2))
    print(f"wrote {out}")


# ---------------------------------------------------------------------- pilot decision


def _gammaincc(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) (Numerical Recipes series / continued fraction)."""
    if x <= 0:
        return 1.0
    lg = math.lgamma(a)
    if x < a + 1:
        term = total = 1.0 / a
        ap = a
        for _ in range(1000):
            ap += 1
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
        return max(0.0, 1.0 - total * math.exp(-x + a * math.log(x) - lg))
    b, c, d = x + 1 - a, 1e300, 1.0 / (x + 1 - a)
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        h *= d * c
        if abs(d * c - 1) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - lg) * h


def kruskal_wallis(groups) -> tuple[float, float]:
    """H statistic (tie-corrected) and its chi-square p-value with len(groups) - 1 degrees of freedom."""
    groups = [np.asarray(g, float) for g in groups if len(g)]
    allv = np.concatenate(groups)
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):  # average ranks over ties
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n, h, start = len(allv), 0.0, 0
    for g in groups:
        r = ranks[start:start + len(g)]
        h += r.sum() ** 2 / len(g)
        start += len(g)
    h = 12 / (n * (n + 1)) * h - 3 * (n + 1)
    _, counts = np.unique(allv, return_counts=True)
    h /= 1 - ((counts ** 3 - counts).sum() / (n ** 3 - n))
    return float(h), _gammaincc((len(groups) - 1) / 2, h / 2)


def go_criteria(npz_path) -> dict:
    """EXPERIMENT-OUTER-LOOP.md section 4 go criteria from a c4-ponder-eval .npz on the tactical set."""
    from .c4data import R_BUCKETS

    f = np.load(npz_path)
    ok, p, r, ply = f["move_ok"], f["p"], f["req_depth"], f["n_moves"]
    K = ok.shape[1]
    stop = (np.cumsum(p, 1) >= 0.5).argmax(1) + 1
    exp_steps = (p * np.arange(1, K + 1)).sum(1)
    at_stop = ok[np.arange(len(stop)), stop - 1]
    deep = r >= 3
    gain = float(at_stop[deep].mean() - ok[deep, 0].mean())
    h, pval = kruskal_wallis([exp_steps[(r >= lo) & (r <= hi)] for lo, hi in R_BUCKETS])
    rhos = [spearman(exp_steps[m], r[m]) for lo, hi in STAGES if (m := (ply >= lo) & (ply <= hi)).sum() > 2]
    rho = float(np.nanmean(rhos))
    acc1, acc2 = float(at_stop[r == 1].mean()), float(at_stop[r == 2].mean())
    c = dict(gain_deep=round(gain, 4), kruskal_h=round(h, 2), kruskal_p=pval, mean_stage_rho=round(rho, 4),
             acc_r1=round(acc1, 4), acc_r2=round(acc2, 4), mean_stop=round(float(stop.mean()), 3))
    c["pass_1_deeper_gain"] = gain >= 0.05
    # p alone is not enough: with ~13k positions an untrained model reaches p < 0.01 at rho ~0.04 (dry run)
    c["pass_2_stop_tracks_r"] = pval < 0.01 and rho >= 0.1
    c["pass_3_learns_game"] = acc1 >= 0.6 and acc2 >= 0.6
    c["passes"] = c["pass_1_deeper_gain"] and c["pass_2_stop_tracks_r"] and c["pass_3_learns_game"]
    return c


def decide_main(argv=None):
    """Score pilot evals; print the winning run name (or 'none') and write a decision JSON."""
    p = argparse.ArgumentParser(description=decide_main.__doc__)
    p.add_argument("--evals", nargs="+", required=True, help="run_name=path/to/eval-tactical-k24.npz")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    rows = {}
    for item in args.evals:
        name, path = item.split("=", 1)
        rows[name] = go_criteria(path) if Path(path).exists() else dict(missing=path, passes=False)
    passing = [n for n, c in rows.items() if c["passes"]]
    winner = max(passing, key=lambda n: rows[n]["mean_stage_rho"]) if passing else "none"
    Path(args.out).write_text(json.dumps(dict(winner=winner, pilots=rows), indent=2, default=float))
    print(winner)


if __name__ == "__main__":
    train_main()
