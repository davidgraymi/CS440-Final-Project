"""Check that our TRM reproduces the official implementation bit-for-bit.

Requires a clone of the official repo and its extra deps:

    git clone https://github.com/SamsungSAILMontreal/TinyRecursiveModels /path/to/TRM
    uv run --with pydantic --with einops python scripts/verify_against_official.py \
        --official-repo /path/to/TRM --ckpt data/ckpt/TRM-MLP-Sudoku-Extreme.pt

Runs both models (fp32, CPU) on the first N Sudoku-Extreme test puzzles for all
16 supervision steps and reports the max logit difference (expected: 0.0).
"""

import argparse
import sys

import torch

from cs440_final_project.data import load_split
from cs440_final_project.evaluate import OFFICIAL_CONFIGS
from cs440_final_project.models import ModelConfig, load_official_trm

p = argparse.ArgumentParser()
p.add_argument("--official-repo", required=True)
p.add_argument("--ckpt", default="data/ckpt/TRM-MLP-Sudoku-Extreme.pt")
p.add_argument("--n", type=int, default=64)
args = p.parse_args()
sys.path.insert(0, args.official_repo)
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1  # noqa: E402

N = args.n
conf = dict(batch_size=N, seq_len=81, puzzle_emb_ndim=512, num_puzzle_identifiers=1, vocab_size=11,
            H_cycles=3, L_cycles=6, H_layers=0, L_layers=2, hidden_size=512, expansion=4, num_heads=8,
            pos_encodings="none", halt_max_steps=16, halt_exploration_prob=0.1, forward_dtype="float32",
            mlp_t=True, puzzle_emb_len=16, no_ACT_continue=True)
sd = torch.load(args.ckpt, map_location="cpu")
official = TinyRecursiveReasoningModel_ACTV1(conf)
official.load_state_dict({k.removeprefix("_orig_mod.model."): v for k, v in sd.items()}, strict=True)
official.eval()
ours = load_official_trm(args.ckpt, ModelConfig(vocab_size=11, seq_len=81, **OFFICIAL_CONFIGS["sudoku"])).eval()

d = load_split("sudoku", "test")
x = torch.from_numpy(d["inputs"][:N].astype("int64") + 1)
y = torch.from_numpy(d["labels"][:N].astype("int64") + 1)
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(N, dtype=torch.int32))
worst = 0.0
with torch.no_grad():
    carry, c2 = official.initial_carry(batch), ours.init_carry(N, "cpu")
    for s in range(16):
        carry, out = official(carry, batch)
        c2, logits, _ = ours.step(x, c2)
        diff = (out["logits"] - logits).abs().max().item()
        worst = max(worst, diff)
        acc_o = (out["logits"].argmax(-1) == y).all(-1).float().mean().item()
        acc_u = (logits.argmax(-1) == y).all(-1).float().mean().item()
        print(f"step {s + 1:2d}  max|dlogit| {diff:.3g}  official acc {acc_o:.4f}  ours acc {acc_u:.4f}")
print("MATCH" if worst < 1e-4 else f"MISMATCH (max diff {worst})")
