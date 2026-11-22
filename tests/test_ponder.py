"""Tests for the outer-loop (PonderNet-style) Connect Four training in c4_ponder.py."""

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from cs440_final_project import c4_ponder as P
from cs440_final_project.c4_train import load
from cs440_final_project.connect4 import INVALID, depth_from_score

ROOT = Path(__file__).resolve().parents[1]
TACTICAL = ROOT / "data" / "c4v2_tactical.npz"
needs_data = pytest.mark.skipif(not TACTICAL.exists(), reason="v2 data not built")


def slow_classes(scores, n_moves):
    out = np.full(scores.shape, P.IGNORE)
    for i in range(len(scores)):
        for c in range(7):
            s = int(scores[i, c])
            if s == INVALID:
                continue
            if s == 0:
                out[i, c] = P.DRAW_CLASS
            else:
                b = int(np.searchsorted(P.DIST_EDGES, depth_from_score(s, int(n_moves[i]))))
                out[i, c] = 8 - b if s > 0 else b
    return out


@needs_data
def test_col_classes_match_solver_scores():
    d = load(TACTICAL)
    sc, n = d["scores"][:3000], d["n_moves"][:3000]
    cls = P.col_classes(sc, n)
    assert np.array_equal(cls, slow_classes(sc, n))
    assert ((cls == P.IGNORE) == (sc == INVALID)).all()
    # classes are ordered like scores: every optimal column has the best class
    opt = d["optimal"][:3000]
    best = np.where(cls == P.IGNORE, -1, cls).max(1, keepdims=True)
    assert (cls[opt] == np.broadcast_to(best, cls.shape)[opt]).all()
    # tactical positions are won, with exactly one fastest (optimal) winning column
    assert (opt.sum(1) == 1).all() and (best[:, 0] >= 5).all()


@needs_data
def test_mirror_flips_every_target():
    d = load(TACTICAL)
    d = {k: v[:200] for k, v in d.items()}
    ds = P.PonderDataset(d, mirror=True)
    ds.rng = np.random.default_rng(0)
    seen = 0
    for i in range(200):
        x, y = (t.numpy() for t in ds[i])
        if np.array_equal(x, d["tokens"][i]):
            continue  # not mirrored this time (or symmetric)
        seen += 1
        msc = d["scores"][i][::-1]
        assert np.array_equal(y[15:22], P.col_classes(msc[None], d["n_moves"][i:i + 1])[0])
        assert np.array_equal(y[:7], d["optimal"][i][::-1])
        assert y[14] == d["value"][i] + 1
    assert seen > 50


def test_halting_and_prior_are_distributions():
    q = torch.randn(5, 7)
    p = P.halting_dist(q)
    assert torch.allclose(p.sum(1), torch.ones(5))
    assert (p >= 0).all()
    prior = P.geometric_prior(7, 3.0, "cpu")
    assert torch.isclose(prior.sum(), torch.tensor(1.0))
    # stopping step: first step with cumulative chance >= 1/2
    p = torch.tensor([[0.1, 0.3, 0.2, 0.4], [0.6, 0.2, 0.1, 0.1]])
    assert P.stop_step(p).tolist() == [3, 1]


def tiny_model(**kw):
    args = Namespace(d_model=32, n_layers=1, n=1, T=1, k_max=4, prior_steps=3.0, **kw)
    return P.build(args, torch.device("cpu"))[1]


def grad_reaching_step1(window, ckpt):
    torch.manual_seed(0)
    model = tiny_model()
    tokens = torch.randint(1, 4, (3, 42))
    grads = {}
    outs = P.episode(model, tokens, 4, window=window, ckpt=ckpt,
                     hook=lambda k, y: y.register_hook(lambda g, k=k: grads.__setitem__(k, g.clone())))
    outs["policy"][3].sum().backward()  # a loss on the last step only
    return grads.get(0), model


def test_window_controls_credit_across_steps():
    g1, _ = grad_reaching_step1(window=1, ckpt=False)
    g0, _ = grad_reaching_step1(window=0, ckpt=False)
    assert g1 is None or g1.abs().max() == 0  # detached: no credit flows back to step 1
    assert g0 is not None and g0.abs().max() > 0


def test_checkpointing_gives_same_gradients():
    _, m_plain = grad_reaching_step1(window=0, ckpt=False)
    _, m_ckpt = grad_reaching_step1(window=0, ckpt=True)
    for (n, a), (_, b) in zip(m_plain.named_parameters(), m_ckpt.named_parameters()):
        if a.grad is None:
            assert b.grad is None, n
        else:
            assert torch.allclose(a.grad, b.grad, atol=1e-6), n


@needs_data
def test_train_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "RUNS_DIR", tmp_path)
    P.train_main(["--run-name", "smoke", "--device", "cpu", "--workers", "0", "--train-limit", "512",
                  "--batch-size", "16", "--k-max", "3", "--steps", "4", "--eval-every", "2", "--eval-limit", "64",
                  "--d-model", "32", "--warmup", "2"])
    log = (tmp_path / "smoke" / "log.jsonl").read_text().splitlines()
    assert any('"eval"' in line for line in log)
    P.eval_main(["--ckpt", str(tmp_path / "smoke" / "ckpt.pt"), "--device", "cpu", "--bootstrap", "20",
                 "--out", str(tmp_path / "e.json")])
    assert (tmp_path / "e.json").exists()


def test_early_stop_of_unroll():
    model = tiny_model()
    with torch.no_grad():
        model.q_head.bias.fill_(8.0)  # stop vote ~1 at every step
    outs = P.episode(model, torch.randint(1, 4, (3, 42)), 6, stop_eps=0.01)
    assert len(outs["q"]) == 1
    outs = P.episode(model, torch.randint(1, 4, (3, 42)), 6, stop_eps=0.0)
    assert len(outs["q"]) == 6


@needs_data
def test_train_smoke_compile_amp_cpu(tmp_path, monkeypatch):
    """--amp and --compile plumbing (fp16 autocast and the scaler are no-ops/disabled paths on CPU)."""
    monkeypatch.setattr(P, "RUNS_DIR", tmp_path)
    P.train_main(["--run-name", "smoke2", "--device", "cpu", "--workers", "0", "--train-limit", "256",
                  "--batch-size", "8", "--k-max", "2", "--steps", "2", "--eval-every", "2", "--eval-limit", "16",
                  "--d-model", "32", "--warmup", "1", "--amp", "fp16"])
    ck = torch.load(tmp_path / "smoke2" / "ckpt.pt", weights_only=True)
    assert "scaler" in ck and ck["step"] == 2


def test_kruskal_wallis_matches_reference():
    # by hand: rank sums 20, 57.5, 42.5 -> H = 7.125; 5 tied pairs -> H / (1 - 30/3360) = 7.1892;
    # 2 degrees of freedom -> p = exp(-H/2)
    h, p = P.kruskal_wallis([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [2, 4, 6, 8, 10]])
    assert abs(h - 7.1892) < 1e-3 and abs(p - np.exp(-7.1892 / 2)) < 1e-4
    h, p = P.kruskal_wallis([np.arange(50), np.arange(50) + 200.0])
    assert p < 1e-10


def test_fixed_steps_loss_grades_last_step_only():
    model = tiny_model()
    tokens = torch.randint(1, 4, (4, 42))
    y = torch.zeros(4, 22, dtype=torch.long)
    y[:, 0] = 1; y[:, 7:14] = 1; y[:, 15:22] = 4
    args = Namespace(value_weight=0.5, col_weight=1.0, beta=0.1, prior_steps=3.0, fixed_steps=3)
    outs = P.episode(model, tokens, 3)
    loss, st = P.ponder_loss(outs, y, args)
    last, _ = P.step_task_loss(outs["policy"][2], outs["value"][2], outs["cols"][2], y, 0.5, 1.0)
    assert torch.isclose(loss, last.mean()) and st["exp_steps"] == 3
