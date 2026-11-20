"""Train and evaluate recursive models on solver-labeled Connect Four positions.

    uv run c4-data                                   # build data/c4_{train,test}.npz first
    uv run c4-train --model trm                      # TRM (recursive thinking model)
    uv run c4-train --model ds-only                  # ablation: deep supervision, no latent z
    uv run c4-train --model direct                   # one-pass transformer baseline
    uv run c4-eval --ckpt runs/<run>/ckpt.pt         # per-step eval + depth analysis
    uv run c4-eval --ckpt runs/<run>/ckpt.pt --steps 32   # think longer than trained

Task: given a position (42 cells from the side to move's view), pick an
optimal column. Targets come from the exact solver: the set of optimal moves
and the game value. Losses:
  * policy: -log P(chosen column is optimal) = -log sum_{c optimal} softmax(policy)_c
  * value:  3-way cross-entropy on loss / draw / win (weight --value-weight)
  * halting head: predicts "the current argmax move is optimal"
Augmentation: left-right mirror (Connect Four is symmetric).

The evaluation records, for every test position and every supervision step,
whether the argmax move is optimal, then reports accuracy per depth bucket
(d = plies to the end under optimal play). The depth hypothesis predicts that
deeper positions need more steps before their accuracy rises.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, Dataset

from .connect4 import DEPTH_BUCKETS, VOCAB_SIZE, bucket_of
from .models import ModelConfig, build_model
from .train import MODEL_PRESETS, ExampleStream, RollingBatch, pick_device, train_step

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "runs"
BUCKET_NAMES = [f"{lo}" if lo == hi else f"{lo}-{hi}" for lo, hi in DEPTH_BUCKETS] + ["draw"]


# ---------------------------------------------------------------------- data


NEEDED = ("tokens", "optimal", "legal", "value", "depth", "n_moves")


def load(path) -> dict:
    """Only the arrays training/eval use (skips e.g. ~160 MB of move strings)."""
    with np.load(path) as f:
        return {k: f[k] for k in NEEDED}


def pack_targets(d: dict) -> np.ndarray:
    """(N, 15) uint8: optimal mask (7), legal mask (7), value class (0 loss, 1 draw, 2 win).
    Kept as uint8 (each DataLoader worker holds a copy); converted per batch."""
    return np.concatenate([d["optimal"], d["legal"], (d["value"] + 1)[:, None]], axis=1).astype(np.uint8)


class C4Dataset(Dataset):
    def __init__(self, d: dict, mirror: bool):
        self.x = np.ascontiguousarray(d["tokens"], dtype=np.uint8)
        self.y = pack_targets(d)
        self.mirror = mirror
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x, y = self.x[i], self.y[i]
        if self.mirror and self.rng.random() < 0.5:
            x = x.reshape(6, 7)[:, ::-1].reshape(-1)
            y = np.concatenate([y[6::-1], y[13:6:-1], y[14:]])
        return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))


def worker_init(worker_id: int):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32 + worker_id)


def game_loss(out, y, value_weight: float):
    optimal = y[:, :7].bool()
    logp = F.log_softmax(out.policy, -1)
    policy_loss = -torch.logsumexp(logp.masked_fill(~optimal, -1e9), -1)
    value_loss = F.cross_entropy(out.value, y[:, 14], reduction="none")
    correct = optimal.gather(1, out.policy.argmax(-1, keepdim=True)).squeeze(1)
    return policy_loss + value_weight * value_loss, correct


# ---------------------------------------------------------------- evaluation


@torch.no_grad()
def run_eval(model, d: dict, device, batch_size: int, steps: int | None = None, verbose: bool = False):
    """Per-position, per-step: move optimal (bool), value correct (bool), halting logit."""
    was_training = model.training
    model.eval()
    steps = steps or model.n_sup
    N = len(d["tokens"])
    move_ok = np.zeros((N, steps), bool)
    value_ok = np.zeros((N, steps), bool)
    qs = np.full((N, steps), np.nan, np.float32)
    x_all = torch.from_numpy(d["tokens"].astype(np.int64))
    y_all = torch.from_numpy(pack_targets(d).astype(np.int64))
    t0 = time.time()
    for i in range(0, N, batch_size):
        x, y = x_all[i:i + batch_size].to(device), y_all[i:i + batch_size].to(device)
        carry = model.init_carry(len(x), device)
        for s in range(steps):
            carry, out, q = model.step(x, carry)
            move_ok[i:i + len(x), s] = y[:, :7].bool().gather(1, out.policy.argmax(-1, keepdim=True)).squeeze(1).cpu().numpy()
            value_ok[i:i + len(x), s] = (out.value.argmax(-1) == y[:, 14]).cpu().numpy()
            if q is not None:
                qs[i:i + len(x), s] = q.float().cpu().numpy()
        if verbose:
            print(f"  {min(i + batch_size, N)}/{N}  move acc {move_ok[:i + len(x), -1].mean():.4f}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    model.train(was_training)
    return move_ok, value_ok, qs


def analyze(path: Path) -> dict:
    """Summarize eval results by depth bucket; writes summary.json and a plot."""
    r = np.load(path)
    move_ok, value_ok, qs, depth, value = r["move_ok"], r["value_ok"], r["q"], r["depth"], r["value"]
    N, steps = move_ok.shape
    stays = np.flip(np.cumprod(np.flip(move_ok, 1), 1), 1).astype(bool)
    solved = stays.any(1)
    steps_needed = np.where(solved, stays.argmax(1) + 1, steps + 1)
    has_q = not np.isnan(qs).all()
    halt = np.where((qs > 0).any(1), (qs > 0).argmax(1) + 1, steps) if has_q else None
    buckets = bucket_of(depth, value)

    def spearman(a, b):
        return float(np.corrcoef(a.argsort().argsort(), b.argsort().argsort())[0, 1])

    rows = []
    for b, name in enumerate(BUCKET_NAMES):
        m = buckets == b
        if not m.any():
            continue
        rows.append(dict(
            depth=name, n=int(m.sum()),
            move_acc=round(float(move_ok[m, -1].mean()), 4),
            value_acc=round(float(value_ok[m, -1].mean()), 4),
            move_acc_by_step=[round(float(v), 4) for v in move_ok[m].mean(0)],
            mean_steps_needed_if_solved=round(float(steps_needed[m & solved].mean()), 3) if (m & solved).any() else None,
            mean_halt_step=round(float(halt[m].mean()), 3) if has_q else None,
        ))
    dec = (value != 0) & solved
    summary = dict(
        source=str(path), n=int(N), steps=int(steps),
        move_acc=round(float(move_ok[:, -1].mean()), 4),
        value_acc=round(float(value_ok[:, -1].mean()), 4),
        move_acc_by_step=[round(float(v), 4) for v in move_ok.mean(0)],
        spearman_depth_vs_steps_needed=round(spearman(depth[dec], steps_needed[dec]), 4) if dec.sum() > 2 else None,
        by_depth=rows,
    )
    if has_q:
        summary["move_acc_at_halt"] = round(float(move_ok[np.arange(N), halt - 1].mean()), 4)
        summary["mean_halt_step"] = round(float(halt.mean()), 3)
        dh = value != 0
        summary["spearman_depth_vs_halt_step"] = round(spearman(depth[dh], halt[dh]), 4)
    path.parent.joinpath("summary.json").write_text(json.dumps(summary, indent=2))
    _plot(rows, steps, path.parent / "move_acc_vs_steps_by_depth.png")
    return summary


def _plot(rows, steps, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(rows)))
    x = np.arange(1, steps + 1)
    for r, c in zip(rows, colors):
        ls = "--" if r["depth"] == "draw" else "-"
        ax.plot(x, r["move_acc_by_step"], ls, marker="o", ms=2.5, color=c, label=f"d={r['depth']} (n={r['n']})")
    ax.set_xlabel("supervision step (thinking iterations)")
    ax.set_ylabel("optimal-move accuracy")
    ax.set_title("Connect Four: accuracy vs. thinking steps, by depth to end")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def balanced_subset(d: dict, per_bucket: int | None, seed: int = 0) -> dict:
    if not per_bucket:
        return d
    rng = np.random.default_rng(seed)
    b = bucket_of(d["depth"], d["value"])
    idx = np.sort(np.concatenate([rng.permutation(np.flatnonzero(b == k))[:per_bucket]
                                  for k in range(len(BUCKET_NAMES))]))
    return {k: v[idx] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(b) else v for k, v in d.items()}


# ------------------------------------------------------------------ training


def train_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODEL_PRESETS, default="trm")
    p.add_argument("--run-name", default=None)
    p.add_argument("--data-prefix", default=str(ROOT / "data" / "c4"))
    p.add_argument("--mixer", choices=["attn", "mlp"], default="mlp")
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int)
    p.add_argument("--n", type=int)
    p.add_argument("--T", type=int)
    p.add_argument("--n-sup", type=int)
    p.add_argument("--steps", type=int, default=50_000, help="optimizer steps")
    p.add_argument("--time-limit-min", type=float, default=0,
                   help="stop after this many minutes of training (0: no limit); evaluates and saves before exiting")
    p.add_argument("--batch-size", type=int, default=768)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--halt-explore", type=float, default=0.1)
    p.add_argument("--value-weight", type=float, default=0.5)
    p.add_argument("--no-mirror", dest="mirror", action="store_false")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp32"], default="auto", help="auto: bf16 on CUDA, else fp32")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--eval-per-bucket", type=int, default=500, help="test positions per depth bucket (0: all)")
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-limit", type=int, default=0, help="use only the first N training positions (smoke tests)")
    # Keeps train.train_step's puzzle-only options defined.
    p.set_defaults(loss="stablemax")
    return p.parse_args(argv)


def train_main(argv=None):
    args = train_args(argv)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    preset = dict(MODEL_PRESETS[args.model])
    kind = preset.pop("kind")
    for k in ("n_layers", "n", "T", "n_sup"):
        if getattr(args, k) is not None:
            preset[k] = getattr(args, k)
    cfg = ModelConfig(vocab_size=VOCAB_SIZE, seq_len=42, d_model=args.d_model, n_heads=args.n_heads,
                      mixer=args.mixer, n_prefix=1, head="game", **preset)
    model = build_model(kind, cfg).to(device)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True)
    if args.compile:
        model.step = torch.compile(model.step)
    use_bf16 = args.dtype == "bf16" or (args.dtype == "auto" and device.type == "cuda")
    autocast = torch.autocast(device.type, dtype=torch.bfloat16) if use_bf16 else nullcontext()

    train = load(f"{args.data_prefix}_train.npz")
    if args.train_limit:
        train = {k: v[:args.train_limit] if isinstance(v, np.ndarray) and v.ndim else v for k, v in train.items()}
    test = balanced_subset(load(f"{args.data_prefix}_test.npz"), args.eval_per_bucket)
    loader = DataLoader(C4Dataset(train, args.mirror), args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.workers, worker_init_fn=worker_init,
                        persistent_workers=args.workers > 0, pin_memory=device.type == "cuda")
    batch = RollingBatch(model, ExampleStream(loader, device), args.batch_size, args.halt_explore)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup)))
    loss_fn = lambda out, y: game_loss(out, y, args.value_weight)  # noqa: E731

    run_name = args.run_name or f"c4-{args.model}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for p in model.parameters())
    meta = dict(args=vars(args), model=asdict(cfg), kind=kind, params=n_params,
                effective_depth=model.effective_depth(), device=str(device), bf16=use_bf16)
    (run_dir / "config.json").write_text(json.dumps(meta, indent=2))
    print(f"{run_name}: {kind} {n_params / 1e6:.2f}M params, depth/step {model.effective_depth()}, "
          f"n_sup {model.n_sup}, train {len(train['tokens'])}, device {device}, bf16 {use_bf16}", flush=True)

    with open(run_dir / "log.jsonl", "a") as log:
        def emit(rec):
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()

        t0, window = time.time(), []
        for step in range(1, args.steps + 1):
            window.append(train_step(model, opt, sched, ema, batch, args, autocast, loss_fn))
            if step % 100 == 0:
                emit(dict(step=step, sec=round(time.time() - t0, 1),
                          **{k: float(np.nanmean([w[k] for w in window])) for k in window[0]}))
                window = []
            out_of_time = args.time_limit_min > 0 and time.time() - t0 > 60 * args.time_limit_min
            if step % args.eval_every == 0 or step == args.steps or out_of_time:
                with autocast:
                    move_ok, value_ok, _ = run_eval(ema.module, test, device, args.eval_batch_size)
                b = bucket_of(test["depth"], test["value"])
                emit(dict(step=step, eval=dict(
                    move_acc=round(float(move_ok[:, -1].mean()), 4),
                    value_acc=round(float(value_ok[:, -1].mean()), 4),
                    move_acc_by_step=[round(float(v), 4) for v in move_ok.mean(0)],
                    move_acc_by_depth={n: round(float(move_ok[b == k, -1].mean()), 4)
                                       for k, n in enumerate(BUCKET_NAMES) if (b == k).any()})))
                torch.save(dict(model=model.state_dict(), ema=ema.module.state_dict(), cfg=asdict(cfg),
                                kind=kind, step=step), run_dir / "ckpt.pt")
            if out_of_time:
                emit(dict(step=step, stopped="time limit", minutes=round((time.time() - t0) / 60, 1)))
                break
        (run_dir / "done.json").write_text(json.dumps(dict(step=step, minutes=round((time.time() - t0) / 60, 1))))


# -------------------------------------------------------------------- c4-eval


def eval_main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", help="checkpoint saved by c4-train")
    src.add_argument("--analyze", help="existing results.npz to re-analyze")
    p.add_argument("--data-prefix", default=str(ROOT / "data" / "c4"))
    p.add_argument("--per-bucket", type=int, default=0, help="test positions per depth bucket (0: all)")
    p.add_argument("--steps", type=int, default=None, help="supervision steps to run (default: trained n_sup)")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    if args.analyze:
        print(json.dumps(analyze(Path(args.analyze)), indent=2))
        return
    device = pick_device(args.device)
    ck = torch.load(args.ckpt, map_location="cpu")
    model = build_model(ck["kind"], ModelConfig(**ck["cfg"]))
    model.load_state_dict(ck["ema"])
    model.to(device)
    test = balanced_subset(load(f"{args.data_prefix}_test.npz"), args.per_bucket)
    steps = args.steps or model.n_sup
    out_dir = Path(args.out) if args.out else (
        RUNS_DIR / f"c4eval-{Path(args.ckpt).parent.name}-steps{steps}-{device.type}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"evaluating {args.ckpt} on {len(test['tokens'])} positions, {steps} steps, device {device}")
    move_ok, value_ok, qs = run_eval(model, test, device, args.batch_size, steps, verbose=True)
    np.savez_compressed(out_dir / "results.npz", move_ok=move_ok, value_ok=value_ok, q=qs,
                        depth=test["depth"], value=test["value"], n_moves=test["n_moves"])
    (out_dir / "eval_config.json").write_text(json.dumps(vars(args), indent=2))
    print(json.dumps(analyze(out_dir / "results.npz"), indent=2))


if __name__ == "__main__":
    train_main()


# ------------------------------------------------------------------- c4-sweep


def sweep_main(argv=None):
    """Test-time thinking sweep: vary TRM's recursions per step (T) and latent
    updates per recursion (n) on a trained checkpoint, without retraining.

        uv run c4-sweep --ckpt runs/c4-trm-overnight/ckpt.pt

    Each setting runs `--steps` supervision steps (default 1) and reports
    optimal-move accuracy per depth bucket against compute, measured as layer
    applications per step: T * (n + 1) * n_layers. If deeper positions gain
    more from extra recursion, the model is using its recursion as lookahead.
    """
    p = argparse.ArgumentParser(description=sweep_main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-prefix", default=str(ROOT / "data" / "c4"))
    p.add_argument("--per-bucket", type=int, default=500)
    p.add_argument("--T", default="1,2,3,4,6", help="comma-separated recursions per step")
    p.add_argument("--n", default="1,2,3,6,9", help="comma-separated latent updates per recursion")
    p.add_argument("--steps", type=int, default=1, help="supervision steps per setting")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    device = pick_device(args.device)
    ck = torch.load(args.ckpt, map_location="cpu")
    if ck["kind"] != "trm":
        raise SystemExit("c4-sweep needs a TRM checkpoint")
    model = build_model("trm", ModelConfig(**ck["cfg"]))
    model.load_state_dict(ck["ema"])
    model.to(device)
    trained = (model.cfg.T, model.cfg.n)
    test = balanced_subset(load(f"{args.data_prefix}_test.npz"), args.per_bucket)
    buckets = bucket_of(test["depth"], test["value"])
    out_dir = Path(args.out) if args.out else RUNS_DIR / f"c4sweep-{Path(args.ckpt).parent.name}-{device.type}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for T in [int(v) for v in args.T.split(",")]:
        for n in [int(v) for v in args.n.split(",")]:
            model.cfg.T, model.cfg.n = T, n
            t0 = time.time()
            move_ok, value_ok, _ = run_eval(model, test, device, args.batch_size, args.steps)
            row = dict(T=T, n=n, layers_per_step=model.effective_depth(), steps=args.steps,
                       move_acc=round(float(move_ok[:, -1].mean()), 4),
                       value_acc=round(float(value_ok[:, -1].mean()), 4),
                       by_depth={name: round(float(move_ok[buckets == k, -1].mean()), 4)
                                 for k, name in enumerate(BUCKET_NAMES) if (buckets == k).any()},
                       sec=round(time.time() - t0, 1))
            rows.append(row)
            print(json.dumps(row), flush=True)
    model.cfg.T, model.cfg.n = trained
    result = dict(ckpt=args.ckpt, trained_T=trained[0], trained_n=trained[1], n_layers=model.cfg.n_layers, n_positions=len(buckets), rows=rows)
    (out_dir / "sweep.json").write_text(json.dumps(result, indent=2))
    _plot_sweep(result, out_dir / "acc_vs_compute_by_depth.png")
    print(f"wrote {out_dir}/sweep.json and acc_vs_compute_by_depth.png")


def _plot_sweep(result, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(result["rows"], key=lambda r: r["layers_per_step"])
    # One point per compute level: the best (T, n) at that compute.
    by_compute = {}
    for r in rows:
        c = r["layers_per_step"]
        if c not in by_compute or r["move_acc"] > by_compute[c]["move_acc"]:
            by_compute[c] = r
    xs = sorted(by_compute)
    names = list(rows[0]["by_depth"])
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(names)))
    for name, c in zip(names, colors):
        ax.plot(xs, [by_compute[x]["by_depth"][name] for x in xs], "--" if name == "draw" else "-",
                marker="o", ms=3, color=c, label=f"d={name}")
    trained = result["trained_T"] * (result["trained_n"] + 1) * result["n_layers"]
    ax.axvline(trained, color="gray", lw=0.8, ls=":")
    ax.text(trained, 0.02, " trained", color="gray", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("layer applications per step (T x (n+1) x layers), log scale")
    ax.set_ylabel("optimal-move accuracy")
    ax.set_title("Test-time thinking: accuracy vs. recursion compute, by depth")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
