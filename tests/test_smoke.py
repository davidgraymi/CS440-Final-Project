from pathlib import Path

import numpy as np
import pytest
import torch

from cs440_final_project.data import dihedral_augment, sudoku_augment
from cs440_final_project.evaluate import OFFICIAL_CONFIGS
from cs440_final_project.models import ModelConfig, build_model, load_official_trm, token_loss
from cs440_final_project.train import main

ROOT = Path(__file__).resolve().parents[1]
SOLVED = np.array([
    int(c) for c in
    "534678912672195348198342567859761423426853791713924856961537284287419635345286179"
])


def valid_sudoku(g):
    g = g.reshape(9, 9)
    groups = [g[i] for i in range(9)] + [g[:, i] for i in range(9)]
    groups += [g[r:r + 3, c:c + 3].ravel() for r in (0, 3, 6) for c in (0, 3, 6)]
    return all(sorted(x) == list(range(1, 10)) for x in groups)


def test_sudoku_augment_preserves_validity_and_givens():
    rng = np.random.default_rng(0)
    q = np.where(rng.random(81) < 0.4, SOLVED, 0)
    assert valid_sudoku(SOLVED)
    for _ in range(50):
        qa, aa = sudoku_augment(q, SOLVED, rng)
        assert valid_sudoku(aa)
        assert np.all((qa == 0) | (qa == aa))  # givens still agree with solution
        assert (qa == 0).sum() == (q == 0).sum()


def test_dihedral_augment_applies_same_transform():
    rng = np.random.default_rng(0)
    q = np.arange(900) % 5
    for _ in range(20):
        qa, aa = dihedral_augment(q, q.copy(), rng, 30)
        assert np.array_equal(qa, aa)
        assert sorted(qa) == sorted(q)


def test_models_forward_backward():
    for kind, mixer, n in [("trm", "mlp", 2), ("trm", "attn", 2), ("trm", "mlp", 0), ("direct", "attn", 0)]:
        cfg = ModelConfig(vocab_size=11, seq_len=81, d_model=32, n_heads=4, mixer=mixer, n=n, T=2, n_sup=3)
        model = build_model(kind, cfg)
        x = torch.randint(1, 11, (4, 81))
        carry = model.init_carry(4, x.device)
        for _ in range(model.n_sup):
            carry, logits, q = model.step(x, carry)
            assert logits.shape == (4, 81, 11)
            loss = token_loss(logits, x).mean()
            if q is not None:
                assert q.shape == (4,)
                loss = loss + q.sum()
            loss.backward()
        assert all(p.grad is not None for p in model.parameters()), kind


def test_official_config_param_count():
    # Official TRM-MLP Sudoku: 5.03M params (2-way q head; ours keeps 1 output).
    cfg = ModelConfig(vocab_size=11, seq_len=81, **OFFICIAL_CONFIGS["sudoku"])
    n = sum(p.numel() for p in build_model("trm", cfg).parameters())
    assert abs(n - 5.03e6) < 0.01e6, n


@pytest.mark.skipif(not (ROOT / "data/ckpt/TRM-MLP-Sudoku-Extreme.pt").exists()
                    or not (ROOT / "data/sudoku_test.npz").exists(), reason="checkpoint/test data not downloaded")
def test_official_checkpoint_solves_puzzles():
    from cs440_final_project.data import load_split

    cfg = ModelConfig(vocab_size=11, seq_len=81, **OFFICIAL_CONFIGS["sudoku"])
    model = load_official_trm(str(ROOT / "data/ckpt/TRM-MLP-Sudoku-Extreme.pt"), cfg).eval()
    d = load_split("sudoku", "test")
    x = torch.from_numpy(d["inputs"][:8].astype(np.int64) + 1)
    y = torch.from_numpy(d["labels"][:8].astype(np.int64) + 1)
    with torch.no_grad():
        carry = model.init_carry(8, "cpu")
        for _ in range(16):
            carry, logits, _ = model.step(x, carry)
    # Reference from the official implementation (fp32, CPU), which ours matches exactly.
    assert (logits.argmax(-1) == y).all(-1).tolist() == [True] * 5 + [False, True, False]


def test_training_loop_runs(tmp_path, monkeypatch):
    import cs440_final_project.train as tr

    monkeypatch.setattr(tr, "RUNS_DIR", tmp_path)
    for model in ("trm", "direct"):
        main(["--task", "sudoku", "--model", model, "--synthetic", "64", "--steps", "6",
              "--batch-size", "16", "--d-model", "32", "--n-heads", "4", "--n-sup", "3",
              "--eval-every", "3", "--eval-batch-size", "32", "--workers", "0", "--device", "cpu",
              "--run-name", model])
        assert (tmp_path / model / "ckpt.pt").exists()
