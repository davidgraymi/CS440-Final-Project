"""Evaluate a TRM checkpoint and analyze thinking steps vs. puzzle difficulty.

    # Reproduce the pretrained-checkpoint sanity check (Sudoku-Extreme):
    uv run trm-eval --official data/ckpt/TRM-MLP-Sudoku-Extreme.pt --num 2000

    # Evaluate a checkpoint from our own training script:
    uv run trm-eval --ckpt runs/<run>/ckpt.pt --num 2000

    # Re-plot / re-summarize saved results without recomputing:
    uv run trm-eval --analyze runs/eval-<name>/results.npz

Runs all n_sup supervision steps on a seeded random sample of the test split
and saves, per puzzle and per step, whether the answer is fully correct and
the halting logit, plus the puzzle's difficulty rating (tdoku backtracks for
Sudoku-Extreme). The analysis answers: do harder puzzles need more thinking
steps before they are solved?

Device note: MPS is fine for development, but the deep recursion amplifies its
floating-point differences from CPU/CUDA, so some answers change (see README
for the measured gap). Report final numbers from CPU or CUDA.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import TASKS, load_split
from .models import ModelConfig, build_model, load_official_trm

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"

# Official TRM-MLP Sudoku-Extreme configuration (all_config.yaml of the checkpoint).
OFFICIAL_CONFIGS = {
    "sudoku": dict(mixer="mlp", d_model=512, n_layers=2, n_heads=8, n=6, T=3, n_sup=16, n_prefix=16),
    "maze": dict(mixer="attn", d_model=512, n_layers=2, n_heads=8, n=6, T=3, n_sup=16, n_prefix=16),
}
RATING_BINS = [0, 1, 10, 25, 50, 10**9]  # Sudoku-Extreme rating edges: [lo, hi)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(args):
    spec = TASKS[args.task]
    if args.official:
        cfg = ModelConfig(vocab_size=spec.vocab_size, seq_len=spec.seq_len, **OFFICIAL_CONFIGS[args.task])
        return load_official_trm(args.official, cfg), "trm"
    ck = torch.load(args.ckpt, map_location="cpu")
    model = build_model(ck["kind"], ModelConfig(**ck["cfg"]))
    model.load_state_dict(ck["ema"])
    return model, ck["kind"]


@torch.no_grad()
def run_eval(model, inputs, labels, device, batch_size, verbose=True):
    """Returns per-puzzle arrays: correct[N, n_sup] (bool), q[N, n_sup] (float)."""
    model.eval().to(device)
    n_sup = model.n_sup
    correct = np.zeros((len(inputs), n_sup), dtype=bool)
    qs = np.full((len(inputs), n_sup), np.nan, dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(inputs), batch_size):
        x = torch.from_numpy(inputs[i:i + batch_size].astype(np.int64) + 1).to(device)
        y = torch.from_numpy(labels[i:i + batch_size].astype(np.int64) + 1).to(device)
        carry = model.init_carry(len(x), device)
        for s in range(n_sup):
            carry, logits, q = model.step(x, carry)
            correct[i:i + len(x), s] = (logits.argmax(-1) == y).all(-1).cpu().numpy()
            if q is not None:
                qs[i:i + len(x), s] = q.float().cpu().numpy()
        done = min(i + batch_size, len(inputs))
        if verbose:
            print(f"  {done}/{len(inputs)}  acc so far {correct[:done, -1].mean():.4f}  "
              f"{time.time() - t0:.0f}s", flush=True)
    model.train()
    return correct, qs


def analyze(path: Path) -> dict:
    """Summarize results.npz and write summary.json + depth_vs_difficulty.png."""
    r = np.load(path)
    correct, qs, ratings = r["correct"], r["q"], r["ratings"]
    N, n_sup = correct.shape

    # Steps needed: first step after which the answer stays correct to the end
    # (n_sup + 1 = never solved).
    stays = np.flip(np.cumprod(np.flip(correct, 1), 1), 1).astype(bool)
    solved = stays[:, 0] | stays.any(1)
    steps_needed = np.where(stays.any(1), stays.argmax(1) + 1, n_sup + 1)
    # Halting step: first step with q > 0 (n_sup if never).
    has_q = not np.isnan(qs).all()
    halt_step = np.where((qs > 0).any(1), (qs > 0).argmax(1) + 1, n_sup) if has_q else None

    def spearman(a, b):
        ra, rb = a.argsort().argsort(), b.argsort().argsort()
        return float(np.corrcoef(ra, rb)[0, 1])

    bins = []
    for lo, hi in zip(RATING_BINS[:-1], RATING_BINS[1:]):
        m = (ratings >= lo) & (ratings < hi)
        if not m.any():
            continue
        sv = m & solved
        bins.append(dict(
            rating=f"[{lo}, {hi if hi < 10**9 else 'inf'})",
            n=int(m.sum()),
            final_acc=round(float(correct[m, -1].mean()), 4),
            acc_by_step=[round(float(v), 4) for v in correct[m].mean(0)],
            mean_steps_needed_if_solved=round(float(steps_needed[sv].mean()), 3) if sv.any() else None,
            mean_halt_step=round(float(halt_step[m].mean()), 3) if has_q else None,
        ))
    summary = dict(
        source=str(path), n=int(N), n_sup=int(n_sup),
        final_exact_acc=round(float(correct[:, -1].mean()), 4),
        acc_by_step=[round(float(v), 4) for v in correct.mean(0)],
        spearman_rating_vs_steps_needed_solved=round(spearman(ratings[solved], steps_needed[solved]), 4),
        by_rating=bins,
    )
    if has_q:
        summary["halted_acc"] = round(float(correct[np.arange(N), halt_step - 1].mean()), 4)
        summary["spearman_rating_vs_halt_step"] = round(spearman(ratings, halt_step), 4)
    out_dir = path.parent
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _plot(bins, n_sup, out_dir / "depth_vs_difficulty.png")
    return summary


def _plot(bins, n_sup, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    steps = np.arange(1, n_sup + 1)
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(bins)))
    for b, c in zip(bins, colors):
        ax.plot(steps, b["acc_by_step"], marker="o", ms=3, color=c, label=f"rating {b['rating']} (n={b['n']})")
    ax.set_xlabel("supervision step (thinking iterations)")
    ax.set_ylabel("exact accuracy")
    ax.set_title("Accuracy vs. thinking steps, by puzzle difficulty")
    ax.set_xticks(steps)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--official", help="official TRM state dict (.pt)")
    src.add_argument("--ckpt", help="checkpoint saved by trm-train")
    src.add_argument("--analyze", help="existing results.npz to re-analyze")
    p.add_argument("--task", choices=TASKS, default="sudoku")
    p.add_argument("--num", type=int, default=2000, help="test puzzles to sample (0: all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=250)
    p.add_argument("--device", default=default_device())
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0: default)")
    p.add_argument("--out", default=None, help="output directory (default runs/eval-<name>)")
    args = p.parse_args(argv)

    if args.analyze:
        print(json.dumps(analyze(Path(args.analyze)), indent=2))
        return
    if args.device == "mps":
        print("note: MPS numerics differ from CPU/CUDA; use --device cpu for final numbers.")
    if args.threads:
        torch.set_num_threads(args.threads)

    test = load_split(args.task, "test")
    idx = np.arange(len(test["inputs"]))
    if args.num and args.num < len(idx):
        idx = np.sort(np.random.default_rng(args.seed).choice(len(idx), args.num, replace=False))
    model, kind = load_model(args)
    name = Path(args.official or args.ckpt).stem if args.official else Path(args.ckpt).parent.name
    out_dir = Path(args.out) if args.out else RUNS_DIR / f"eval-{name}-n{len(idx)}-s{args.seed}-{args.device}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"evaluating {name} ({kind}) on {len(idx)} {args.task} test puzzles, device {args.device}")

    correct, qs = run_eval(model, test["inputs"][idx], test["labels"][idx], torch.device(args.device), args.batch_size)
    np.savez_compressed(out_dir / "results.npz", correct=correct, q=qs, ratings=test["ratings"][idx], index=idx)
    (out_dir / "eval_config.json").write_text(json.dumps(vars(args), indent=2))
    print(json.dumps(analyze(out_dir / "results.npz"), indent=2))


if __name__ == "__main__":
    main()
