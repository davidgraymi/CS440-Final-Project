"""Connect Four (7x6): game logic, exact solver labels, and dataset generation.

Labels come from Pascal Pons' solver (github.com/PascalPons/connect4, AGPL-3.0),
run as a separate process; install it with `scripts/setup_connect4_solver.sh`.

    uv run c4-data --num 200000 --out data/c4_pilot.npz      # generate + label

Solver scores (from the side to move, Pons' convention): 0 is a draw; a
positive score s means the side to move wins, and larger s means sooner. With
n stones on the board and optimal play by both sides (the winner wins as fast
as possible, the loser delays as long as possible):

    win  (s > 0):  d = 2 * ((43 - n) // 2 - s) + 1   plies to the end
    loss (s < 0):  d = 2 * ((42 - n) // 2 + s) + 2
    draw (s = 0):  d = 42 - n                          (board fills up)

`d` is the depth label for the depth-scaling hypothesis.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

W, H = 7, 6
ROOT = Path(__file__).resolve().parents[2]
SOLVER_DIR = Path(os.environ.get("C4_SOLVER_DIR", ROOT / "third_party" / "connect4"))
INVALID = -1000  # solver output for a full column

# Model tokens (0 is PAD, matching the puzzle tasks): empty / side to move / opponent.
EMPTY, OWN, OPP = 1, 2, 3
VOCAB_SIZE = 4


# --------------------------------------------------------------- game logic


class Board:
    """Minimal Connect Four state. grid[r, c], r = 0 is the bottom row;
    1 = first player, 2 = second player."""

    def __init__(self):
        self.grid = np.zeros((H, W), dtype=np.int8)
        self.heights = [0] * W
        self.moves: list[int] = []

    @property
    def n(self) -> int:
        return len(self.moves)

    @property
    def to_move(self) -> int:
        return 1 + self.n % 2

    def legal(self) -> list[int]:
        return [c for c in range(W) if self.heights[c] < H]

    def is_winning_move(self, c: int) -> bool:
        """Would the side to move connect four by playing column c?"""
        r, p, g = self.heights[c], self.to_move, self.grid
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            for sgn in (1, -1):
                rr, cc = r + sgn * dr, c + sgn * dc
                while 0 <= rr < H and 0 <= cc < W and g[rr, cc] == p:
                    count += 1
                    rr, cc = rr + sgn * dr, cc + sgn * dc
            if count >= 4:
                return True
        return False

    def play(self, c: int):
        self.grid[self.heights[c], c] = self.to_move
        self.heights[c] += 1
        self.moves.append(c)

    def move_string(self) -> str:
        """Pons' solver format: 1-indexed columns."""
        return "".join(str(c + 1) for c in self.moves)

    def key(self) -> bytes:
        return self.grid.tobytes()

    def tokens(self) -> np.ndarray:
        """42 tokens from the side to move's perspective, top row first."""
        g = np.flipud(self.grid).reshape(-1)
        out = np.full(W * H, EMPTY, dtype=np.uint8)
        out[g == self.to_move] = OWN
        out[(g != 0) & (g != self.to_move)] = OPP
        return out


def board_from_moves(moves: str) -> Board:
    b = Board()
    for ch in moves:
        b.play(int(ch) - 1)
    return b


def depth_from_score(score: int, n: int) -> int:
    """Plies until the game ends under optimal play (see module docstring)."""
    if score > 0:
        return 2 * ((43 - n) // 2 - score) + 1
    if score < 0:
        return 2 * ((42 - n) // 2 + score) + 2
    return 42 - n


# ------------------------------------------------------------------ solver


class Solver:
    """Persistent `c4solver -a` process: per-column scores for many positions."""

    def __init__(self, solver_dir: Path = SOLVER_DIR):
        exe, book = solver_dir / "c4solver", solver_dir / "7x6.book"
        if not exe.exists():
            raise FileNotFoundError(f"{exe} missing; run scripts/setup_connect4_solver.sh")
        self.proc = subprocess.Popen([str(exe), "-a", "-b", str(book)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def analyze(self, move_strings: list[str]) -> np.ndarray:
        """Scores (N, 7) for the side to move; INVALID for full columns.
        Positions must be non-terminal (the solver skips terminal ones)."""
        out = np.empty((len(move_strings), W), dtype=np.int16)
        for i, ms in enumerate(move_strings):
            self.proc.stdin.write(ms + "\n")
            self.proc.stdin.flush()
            parts = self.proc.stdout.readline().split()
            # The empty position prints no move string, so take the last 7 fields.
            assert len(parts) >= W and (parts[0] == ms or ms == ""), (ms, parts)
            out[i] = [int(x) for x in parts[-W:]]
        return out

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


_worker_solver: Solver | None = None


def _solve_chunk(move_strings: list[str]) -> np.ndarray:
    global _worker_solver
    if _worker_solver is None:
        _worker_solver = Solver()
    return _worker_solver.analyze(move_strings)


def solve_parallel(move_strings: list[str], workers: int, chunk: int = 200) -> np.ndarray:
    chunks = [move_strings[i:i + chunk] for i in range(0, len(move_strings), chunk)]
    with ProcessPoolExecutor(workers) as ex:
        return np.concatenate(list(ex.map(_solve_chunk, chunks)))


# ---------------------------------------------------------------- sampling


def opponent_can_win_after(b: Board, c: int) -> bool:
    b.play(c)
    threat = any(b.is_winning_move(x) for x in b.legal())
    b.grid[b.heights[c] - 1, c] = 0
    b.heights[c] -= 1
    b.moves.pop()
    return threat


def random_position(rng: np.random.Generator, n_target: int, safe: bool = False) -> Board | None:
    """Play n_target random moves that do not end the game.

    safe=False: uniformly random. Positions then usually contain an unblocked
    threat, so most have depth 1.
    safe=True: random among moves that leave the opponent no immediate win
    (when such moves exist), i.e. players block threats and avoid handing over
    wins. Gives realistic positions with longer forced sequences.

    Returns None if the game cannot continue without ending."""
    b = Board()
    while b.n < n_target:
        options = [c for c in b.legal() if not b.is_winning_move(c)]
        if not options:
            return None
        if safe:
            options = [c for c in options if not opponent_can_win_after(b, c)] or options
        b.play(int(rng.choice(options)))
    if not b.legal():
        return None
    return b


def _sample_chunk(args) -> list[tuple[bytes, str]]:
    num, seed, min_ply, max_ply, safe_frac = args
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < num:
        b = random_position(rng, int(rng.integers(min_ply, max_ply + 1)), safe=rng.random() < safe_frac)
        if b is not None:
            out.append((b.key(), b.move_string()))
    return out


def sample_positions(num: int, seed: int, workers: int, min_ply: int = 0, max_ply: int = 41,
                     safe_frac: float = 0.8) -> list[str]:
    """Move strings of `num` unique non-terminal positions, move counts uniform
    in [min_ply, max_ply]. A `safe_frac` fraction come from safe-random play,
    the rest from uniform play. Deterministic given (seed, workers)."""
    seen, out, round_ = set(), [], 0
    with ProcessPoolExecutor(workers) as ex:
        while len(out) < num:
            need = int((num - len(out)) * 1.2) + 100
            jobs = [(need // workers + 1, (seed, round_, w), min_ply, max_ply, safe_frac) for w in range(workers)]
            for chunk in ex.map(_sample_chunk, jobs):
                for key, ms in chunk:
                    if key not in seen and len(out) < num:
                        seen.add(key)
                        out.append(ms)
            round_ += 1
    return out


def label(move_strings: list[str], scores: np.ndarray) -> dict:
    boards = [board_from_moves(ms) for ms in move_strings]
    n = np.array([b.n for b in boards], dtype=np.int16)
    valid = scores != INVALID
    best = np.where(valid, scores, -10**4).max(1)
    return dict(
        tokens=np.stack([b.tokens() for b in boards]),
        moves=np.array(move_strings),
        n_moves=n,
        scores=scores.astype(np.int16),          # per column, side to move; INVALID if full
        legal=valid,
        optimal=valid & (scores == best[:, None]),  # all optimal moves
        value=np.sign(best).astype(np.int8),     # +1 win / 0 draw / -1 loss for side to move
        score=best.astype(np.int16),
        depth=np.array([depth_from_score(int(s), int(k)) for s, k in zip(best, n)], dtype=np.int16),
    )


# Depth buckets for the balanced test set: decisive depths [lo, hi], then draws.
DEPTH_BUCKETS = [(1, 1), (2, 2), (3, 4), (5, 6), (7, 8), (9, 12), (13, 16), (17, 24), (25, 42)]


def bucket_of(depth: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Bucket index per position; len(DEPTH_BUCKETS) for draws."""
    b = np.full(len(depth), len(DEPTH_BUCKETS))
    for i, (lo, hi) in enumerate(DEPTH_BUCKETS):
        b[(value != 0) & (depth >= lo) & (depth <= hi)] = i
    return b


def split_balanced(data: dict, per_bucket: int, seed: int) -> tuple[dict, dict]:
    """Test set: up to `per_bucket` positions from each depth bucket; train: the rest."""
    rng = np.random.default_rng(seed)
    buckets = bucket_of(data["depth"], data["value"])
    test_idx = np.concatenate([
        rng.permutation(np.flatnonzero(buckets == b))[:per_bucket] for b in range(len(DEPTH_BUCKETS) + 1)
    ])
    test_mask = np.zeros(len(buckets), bool)
    test_mask[test_idx] = True
    return ({k: v[~test_mask] for k, v in data.items()}, {k: v[test_mask] for k, v in data.items()})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num", type=int, default=1_000_000, help="total positions (train + test)")
    p.add_argument("--test-per-bucket", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-ply", type=int, default=0)
    p.add_argument("--max-ply", type=int, default=41)
    p.add_argument("--safe-frac", type=float, default=0.8, help="fraction of positions from safe-random play")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--out-prefix", default=str(ROOT / "data" / "c4"), help="writes <prefix>_train.npz, <prefix>_test.npz")
    args = p.parse_args(argv)

    t0 = time.time()
    moves = sample_positions(args.num, args.seed, args.workers, args.min_ply, args.max_ply, args.safe_frac)
    print(f"sampled {len(moves)} unique positions in {time.time() - t0:.1f}s", flush=True)
    t0 = time.time()
    data = label(moves, solve_parallel(moves, args.workers))
    print(f"solved and labeled in {time.time() - t0:.1f}s", flush=True)

    train, test = split_balanced(data, args.test_per_bucket, args.seed)
    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    meta = dict(seed=args.seed, safe_frac=args.safe_frac, min_ply=args.min_ply, max_ply=args.max_ply)
    for name, d in (("train", train), ("test", test)):
        np.savez_compressed(f"{args.out_prefix}_{name}.npz", **d, **meta)
        v, dep = d["value"], d["depth"]
        counts = np.bincount(bucket_of(dep, v), minlength=len(DEPTH_BUCKETS) + 1)
        print(f"{name}: {len(v)} positions | win {(v == 1).mean():.3f} draw {(v == 0).mean():.3f} "
              f"loss {(v == -1).mean():.3f}")
        print("  per depth bucket " + " ".join(f"{lo}-{hi}:{c}" for (lo, hi), c in zip(DEPTH_BUCKETS, counts))
              + f" draw:{counts[-1]}")


if __name__ == "__main__":
    main()
