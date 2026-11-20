"""Sanity-check datasets: Sudoku-Extreme and Maze-Hard (30x30).

Both come from the HRM authors on Hugging Face (`sapientinc/...`), the same
data TRM reports on. Each is a CSV with columns: source, question, answer, rating.

`load_split(task, split)` converts a CSV to int arrays and caches it as
`data/<task>_<split>.npz`. It reads `data/<task>_<split>.csv` if present,
otherwise downloads it from Hugging Face. `PuzzleDataset` serves tensors,
applying random symmetry augmentation to training examples on the fly.

Token ids follow the official TRM/HRM encoding: 0 is PAD and every raw symbol
is shifted by +1 (Sudoku: blank=1, digits 1-9 -> 2-10).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class TaskSpec:
    repo: str
    seq_len: int
    vocab_size: int
    train_subsample: int | None  # TRM/HRM train on a 1k-example subset


TASKS = {
    "sudoku": TaskSpec("sapientinc/sudoku-extreme", 81, 11, 1000),
    "maze": TaskSpec("sapientinc/maze-30x30-hard-1k", 900, 6, None),
}

MAZE_CHARSET = "# SGo"  # wall, open, start, goal, path


def _encode(s: str, task: str) -> np.ndarray:
    if task == "sudoku":
        return np.frombuffer(s.replace(".", "0").encode(), dtype=np.uint8) - ord("0")
    lut = {c: i for i, c in enumerate(MAZE_CHARSET)}
    return np.array([lut[c] for c in s], dtype=np.uint8)


def _read_csv(path: Path, task: str, seq_len: int):
    qs, ans, ratings = [], [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for _source, q, a, rating in reader:
            assert len(q) == len(a) == seq_len, (len(q), len(a))
            qs.append(_encode(q, task))
            ans.append(_encode(a, task))
            ratings.append(int(rating))
    return np.stack(qs), np.stack(ans), np.array(ratings, dtype=np.int32)


def load_split(task: str, split: str, seed: int = 0) -> dict:
    """Return {"inputs", "labels", "ratings"} arrays (raw symbols, no +1 offset)."""
    spec = TASKS[task]
    cache = DATA_DIR / f"{task}_{split}.npz"
    if not cache.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = DATA_DIR / f"{task}_{split}.csv"
        if not csv_path.exists():
            from huggingface_hub import hf_hub_download

            csv_path = Path(hf_hub_download(spec.repo, f"{split}.csv", repo_type="dataset"))
        q, a, r = _read_csv(csv_path, task, spec.seq_len)
        if split == "train" and spec.train_subsample is not None:
            idx = np.random.default_rng(seed).choice(len(q), spec.train_subsample, replace=False)
            q, a, r = q[idx], a[idx], r[idx]
        np.savez_compressed(cache, inputs=q, labels=a, ratings=r)
    return dict(np.load(cache))


# ---------------------------------------------------------------- augmentation


def sudoku_augment(q: np.ndarray, a: np.ndarray, rng: np.random.Generator):
    """Validity-preserving Sudoku symmetries: digit relabeling, transpose,
    band/stack permutations, and row/column permutations within them."""
    digits = np.concatenate([[0], rng.permutation(9) + 1])

    def axis_perm():
        bands = rng.permutation(3)
        return np.concatenate([b * 3 + rng.permutation(3) for b in bands])

    rows, cols = axis_perm(), axis_perm()
    transpose = rng.random() < 0.5

    def apply(g):
        g = g.reshape(9, 9)
        if transpose:
            g = g.T
        return digits[g[rows][:, cols]].reshape(-1)

    return apply(q), apply(a)


def dihedral_augment(q: np.ndarray, a: np.ndarray, rng: np.random.Generator, side: int):
    """One of the 8 rotations/reflections of a square grid."""
    k, flip = rng.integers(4), rng.random() < 0.5

    def apply(g):
        g = np.rot90(g.reshape(side, side), k)
        return (np.fliplr(g) if flip else g).reshape(-1).copy()

    return apply(q), apply(a)


# --------------------------------------------------------------------- dataset


class PuzzleDataset(Dataset):
    def __init__(
        self,
        task: str,
        split: str,
        augment: bool = False,
        epoch_multiplier: int = 1,
        limit: int | None = None,
        arrays: dict | None = None,
    ):
        """`epoch_multiplier` repeats the (augmented) training set so an
        "epoch" covers many random symmetries of the same 1k puzzles.
        `arrays` ({"inputs", "labels"}) lets tests inject synthetic data."""
        self.task = task
        if arrays is None:
            arrays = load_split(task, split)
        self.inputs = arrays["inputs"][:limit]
        self.labels = arrays["labels"][:limit]
        self.augment = augment
        self.mult = epoch_multiplier
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.inputs) * self.mult

    def __getitem__(self, i):
        i %= len(self.inputs)
        q, a = self.inputs[i], self.labels[i]
        if self.augment:
            if self.task == "sudoku":
                q, a = sudoku_augment(q, a, self.rng)
            else:
                q, a = dihedral_augment(q, a, self.rng, 30)
        return torch.from_numpy(q.astype(np.int64) + 1), torch.from_numpy(a.astype(np.int64) + 1)


def worker_init(worker_id: int):
    """Give each DataLoader worker its own augmentation RNG stream."""
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32 + worker_id)
