"""Train and evaluate models on the sanity-check puzzles.

    uv run trm-train --task sudoku --model trm            # TRM-MLP (paper setup)
    uv run trm-train --task sudoku --model ds-only        # deep supervision, no latent z
    uv run trm-train --task sudoku --model direct         # one-pass transformer
    uv run trm-train --task maze   --model trm            # TRM-Att on Maze-Hard
    uv run trm-train --task sudoku --model trm --synthetic 256 --steps 20   # smoke test

Training follows the official TRM loop. A batch of B "slots" is carried
across optimizer steps; each step runs one supervision step for every slot
and updates the weights. A slot halts when its halting head fires (q > 0,
with 10% random exploration of a minimum step count) or after n_sup steps,
and is then refilled with a fresh training example. One optimizer step =
one supervision step over B examples.

Defaults reproduce the TRM paper / checkpoint config for Sudoku-Extreme:
batch 768, lr 1e-4, weight decay 1.0, warmup 2000, EMA 0.999, bf16 on CUDA.
The official run used ~65k optimizer steps (50k epochs of 1k puzzles).
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
from torch.utils.data import DataLoader

from .data import TASKS, PuzzleDataset, load_split, worker_init
from .evaluate import run_eval
from .models import ModelConfig, build_model, token_loss

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"

# Sudoku uses the MLP mixer (attention-free), Maze uses self-attention (TRM paper).
TASK_DEFAULTS = {
    "sudoku": dict(mixer="mlp", eval_limit=10_000),
    "maze": dict(mixer="attn", eval_limit=None),
}
MODEL_PRESETS = {
    "trm": dict(kind="trm", n_layers=2, n=6, T=3, n_sup=16),
    "ds-only": dict(kind="trm", n_layers=2, n=0, T=3, n_sup=16),
    "direct": dict(kind="direct", n_layers=8, n=0, T=1, n_sup=1),
}


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synthetic_split(task: str, n: int, seed: int) -> dict:
    """Random token grids with the task's shapes, for smoke tests only."""
    spec = TASKS[task]
    rng = np.random.default_rng(seed)
    hi = spec.vocab_size - 1  # raw symbols are 0..vocab-2 (+1 offset added later)
    a = rng.integers(1 if task == "sudoku" else 0, hi, (n, spec.seq_len))
    q = np.where(rng.random(a.shape) < 0.5, a, 0) if task == "sudoku" else a.copy()
    return dict(inputs=q.astype(np.uint8), labels=a.astype(np.uint8), ratings=np.zeros(n, np.int32))


class ExampleStream:
    """Endless supply of training examples, handed out n at a time."""

    def __init__(self, loader, device):
        self.loader, self.device = loader, device
        self.it = iter(loader)
        self.buf_x = self.buf_y = None

    def take(self, n: int):
        while self.buf_x is None or len(self.buf_x) < n:
            try:
                x, y = next(self.it)
            except StopIteration:
                self.it = iter(self.loader)
                continue
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            self.buf_x = x if self.buf_x is None else torch.cat((self.buf_x, x))
            self.buf_y = y if self.buf_y is None else torch.cat((self.buf_y, y))
        x, y = self.buf_x[:n], self.buf_y[:n]
        self.buf_x, self.buf_y = self.buf_x[n:], self.buf_y[n:]
        return x, y


class RollingBatch:
    """B slots, each holding one example mid-way through its supervision steps."""

    def __init__(self, model, stream: ExampleStream, batch_size: int, explore: float):
        self.model, self.stream, self.B, self.explore = model, stream, batch_size, explore
        dev = stream.device
        self.x, self.y = stream.take(batch_size)
        self.carry = model.init_carry(batch_size, dev)
        self.steps = torch.zeros(batch_size, dtype=torch.long, device=dev)
        self.halted = torch.zeros(batch_size, dtype=torch.bool, device=dev)

    def refill(self):
        idx = self.halted.nonzero().squeeze(-1)
        if len(idx) == 0:
            return
        x, y = self.stream.take(len(idx))
        self.x[idx], self.y[idx] = x, y
        self.steps[idx] = 0
        if self.carry is not None:
            fresh = self.model.init_carry(len(idx), self.x.device)
            for c, f in zip(self.carry, fresh):
                c[idx] = f

    def advance(self, q):
        """Update step counts and decide which slots halt after this step."""
        n_sup = self.model.n_sup
        self.steps += 1
        halted = self.steps >= n_sup
        if q is not None and n_sup > 1:
            halted |= q.detach() > 0
            explore = torch.rand_like(q) < self.explore
            min_steps = explore * torch.randint_like(self.steps, 2, n_sup + 1)
            halted &= self.steps >= min_steps
        self.halted = halted


def train_step(model, opt, sched, ema, batch: RollingBatch, args, autocast) -> dict:
    batch.refill()
    with autocast:
        batch.carry, logits, q = model.step(batch.x, batch.carry)
        correct = (logits.argmax(-1) == batch.y).all(-1)
        lm = token_loss(logits, batch.y, args.loss)
        loss = lm
        if q is not None:
            loss = loss + 0.5 * F.binary_cross_entropy_with_logits(q, correct.float(), reduction="none")
        loss = loss.mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    opt.step()
    sched.step()
    ema.update_parameters(model)
    batch.advance(q)
    h = batch.halted
    return dict(loss=loss.item(), lm=lm.mean().item(),
                exact_at_halt=correct[h].float().mean().item() if h.any() else float("nan"),
                steps_at_halt=batch.steps[h].float().mean().item() if h.any() else float("nan"))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=TASKS, default="sudoku")
    p.add_argument("--model", choices=MODEL_PRESETS, default="trm")
    p.add_argument("--run-name", default=None)
    # architecture overrides (default: preset / task default)
    p.add_argument("--mixer", choices=["attn", "mlp"])
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int)
    p.add_argument("--n", type=int)
    p.add_argument("--T", type=int)
    p.add_argument("--n-sup", type=int)
    # optimization (TRM paper values)
    p.add_argument("--steps", type=int, default=65_000, help="optimizer steps")
    p.add_argument("--batch-size", type=int, default=768)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--grad-clip", type=float, default=0.0, help="0 disables (official: none)")
    p.add_argument("--halt-explore", type=float, default=0.1)
    p.add_argument("--loss", choices=["stablemax", "softmax"], default="stablemax")
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp32"], default="auto", help="auto: bf16 on CUDA, else fp32")
    p.add_argument("--compile", action="store_true", help="torch.compile the model (CUDA)")
    # evaluation / infra
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--eval-limit", type=int, default=-1, help="test puzzles per eval (-1: task default, 0: all)")
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--synthetic", type=int, default=0, help="use N random examples instead of real data")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    spec, tdef, preset = TASKS[args.task], TASK_DEFAULTS[args.task], dict(MODEL_PRESETS[args.model])
    kind = preset.pop("kind")
    for k in ("n_layers", "n", "T", "n_sup"):
        if getattr(args, k) is not None:
            preset[k] = getattr(args, k)
    cfg = ModelConfig(vocab_size=spec.vocab_size, seq_len=spec.seq_len, d_model=args.d_model,
                      n_heads=args.n_heads, mixer=args.mixer or tdef["mixer"], **preset)
    model = build_model(kind, cfg).to(device)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True)
    if args.compile:  # after the EMA copy, so only the training model is compiled
        model.step = torch.compile(model.step)

    use_bf16 = args.dtype == "bf16" or (args.dtype == "auto" and device.type == "cuda")
    autocast = torch.autocast(device.type, dtype=torch.bfloat16) if use_bf16 else nullcontext()

    if args.synthetic:
        train_arr, test_arr = synthetic_split(args.task, args.synthetic, 0), synthetic_split(args.task, args.synthetic, 1)
    else:
        train_arr, test_arr = load_split(args.task, "train"), load_split(args.task, "test")
    eval_limit = tdef["eval_limit"] if args.eval_limit < 0 else (args.eval_limit or None)
    eval_idx = np.arange(len(test_arr["inputs"]))
    if eval_limit and eval_limit < len(eval_idx):  # same seeded sample as trm-eval
        eval_idx = np.sort(np.random.default_rng(0).choice(len(eval_idx), eval_limit, replace=False))

    train_ds = PuzzleDataset(args.task, "train", augment=args.augment, epoch_multiplier=1000, arrays=train_arr)
    loader = DataLoader(train_ds, args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers,
                        worker_init_fn=worker_init, persistent_workers=args.workers > 0,
                        pin_memory=device.type == "cuda")
    batch = RollingBatch(model, ExampleStream(loader, device), args.batch_size, args.halt_explore)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup)))

    run_name = args.run_name or f"{args.task}-{args.model}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for p in model.parameters())
    meta = dict(args=vars(args), model=asdict(cfg), kind=kind, params=n_params,
                effective_depth=model.effective_depth(), device=str(device), bf16=use_bf16)
    (run_dir / "config.json").write_text(json.dumps(meta, indent=2))
    print(f"{run_name}: {kind} {n_params / 1e6:.2f}M params, depth/step {model.effective_depth()}, "
          f"n_sup {model.n_sup}, device {device}, bf16 {use_bf16}")

    with open(run_dir / "log.jsonl", "a") as log:
        def emit(rec):
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()

        t0, window = time.time(), []
        for step in range(1, args.steps + 1):
            window.append(train_step(model, opt, sched, ema, batch, args, autocast))
            if step % 100 == 0:
                emit(dict(step=step, sec=round(time.time() - t0, 1),
                          **{k: float(np.nanmean([w[k] for w in window])) for k in window[0]}))
                window = []
            if step % args.eval_every == 0 or step == args.steps:
                with autocast:
                    correct, _ = run_eval(ema.module, test_arr["inputs"][eval_idx], test_arr["labels"][eval_idx],
                                          device, args.eval_batch_size, verbose=False)
                emit(dict(step=step, eval=dict(n=len(eval_idx), exact=float(correct[:, -1].mean()),
                                               exact_by_step=[round(float(v), 4) for v in correct.mean(0)])))
                torch.save(dict(model=model.state_dict(), ema=ema.module.state_dict(), cfg=asdict(cfg),
                                kind=kind, step=step), run_dir / "ckpt.pt")


if __name__ == "__main__":
    main()
