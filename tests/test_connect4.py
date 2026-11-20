from functools import lru_cache

import numpy as np
import torch
import pytest

from cs440_final_project.connect4 import (
    EMPTY, OPP, OWN, SOLVER_DIR, Board, board_from_moves, depth_from_score, label, random_position,
)


def brute_force(moves: str) -> tuple[int, int]:
    """Exact (value, plies to end) by exhaustive search: the winner wins as
    fast as possible, the loser delays as long as possible."""

    @lru_cache(maxsize=None)
    def solve(ms: str) -> tuple[int, int]:
        b = board_from_moves(ms)
        legal = b.legal()
        if not legal:
            return 0, 0
        if any(b.is_winning_move(c) for c in legal):
            return 1, 1
        best = None
        for c in legal:
            v, d = solve(ms + str(c + 1))
            cand = (-v, d + 1)
            if best is None or _better(cand, best):
                best = cand
        return best

    return solve(moves)


def _better(a, b):
    """Preference order for the side to move: win (sooner) > draw > loss (later)."""
    if a[0] != b[0]:
        return a[0] > b[0]
    return a[1] < b[1] if a[0] == 1 else a[1] > b[1]


def test_tokens_perspective():
    b = board_from_moves("44")
    t = b.tokens().reshape(6, 7)  # top row first
    assert t[-1, 3] == OWN and t[-2, 3] == OPP  # first player to move again
    assert (t == EMPTY).sum() == 40
    b.play(0)
    t = b.tokens().reshape(6, 7)
    assert t[-1, 3] == OPP and t[-1, 0] == OPP and t[-2, 3] == OWN


def test_winning_move_detection():
    b = board_from_moves("121212")  # first player has three in column 1
    assert b.is_winning_move(0)
    assert not b.is_winning_move(2)


@pytest.mark.skipif(not (SOLVER_DIR / "c4solver").exists(), reason="solver not installed")
def test_depth_formula_matches_brute_force():
    from cs440_final_project.connect4 import Solver

    rng = np.random.default_rng(0)
    boards = []
    while len(boards) < 60:
        b = random_position(rng, int(rng.integers(32, 38)))
        if b is not None:
            boards.append(b)
    solver = Solver()
    ms = [b.move_string() for b in boards]
    data = label(ms, solver.analyze(ms))
    solver.close()
    for b, v, d in zip(boards, data["value"], data["depth"]):
        assert brute_force(b.move_string()) == (v, d), b.move_string()
    assert len(set(data["value"].tolist())) > 1  # sample covers more than one outcome


def test_depth_formula_examples():
    # From the solver: after "4455" the side to move wins in 3 plies (score 18).
    assert depth_from_score(18, 4) == 3
    # After "12121" the side to move loses immediately unless it blocks (score -18).
    assert depth_from_score(-18, 5) == 2


def _fake_dataset(n: int, seed: int = 0) -> dict:
    """Random positions with made-up labels (no solver needed)."""
    from cs440_final_project.connect4 import label

    rng = np.random.default_rng(seed)
    moves = []
    while len(moves) < n:
        b = random_position(rng, int(rng.integers(0, 30)), safe=True)
        if b is not None:
            moves.append(b.move_string())
    legal = np.stack([[board_from_moves(m).heights[c] < 6 for c in range(7)] for m in moves])
    scores = np.where(legal, rng.integers(-5, 6, (n, 7)), -1000)
    return label(moves, scores)


def test_mirror_augmentation_is_consistent():
    from cs440_final_project.c4_train import C4Dataset

    d = _fake_dataset(50)
    ds = C4Dataset(d, mirror=True)
    for i in range(50):
        for _ in range(4):
            x, y = ds[i]
            top = x.view(6, 7)[0]
            assert torch.equal(y[7:14].bool(), top == EMPTY)  # legal mask matches mirrored board
            assert y[:7][~y[7:14].bool()].sum() == 0  # optimal moves are legal


def test_c4_train_and_eval_smoke(tmp_path, monkeypatch):
    import cs440_final_project.c4_train as ct

    d = _fake_dataset(200)
    d["depth"] = np.random.default_rng(0).integers(1, 30, len(d["depth"])).astype(np.int16)
    for split in ("train", "test"):
        np.savez(tmp_path / f"c4_{split}.npz", **d)
    monkeypatch.setattr(ct, "RUNS_DIR", tmp_path / "runs")
    for model in ("trm", "direct"):
        ct.train_main(["--model", model, "--data-prefix", str(tmp_path / "c4"), "--steps", "4",
                       "--batch-size", "16", "--d-model", "32", "--n-heads", "4", "--n-sup", "3",
                       "--eval-every", "2", "--eval-per-bucket", "5", "--workers", "0", "--device", "cpu",
                       "--run-name", model])
        ckpt = tmp_path / "runs" / model / "ckpt.pt"
        assert ckpt.exists()
        ct.eval_main(["--ckpt", str(ckpt), "--data-prefix", str(tmp_path / "c4"), "--per-bucket", "5",
                      "--steps", "4" if model == "trm" else "1", "--device", "cpu",
                      "--out", str(tmp_path / f"eval-{model}")])
        assert (tmp_path / f"eval-{model}" / "summary.json").exists()
