"""Connect Four dataset v2 for the depth-hypothesis experiment (EXPERIMENT.md, section 3).

Two steps, so the slow part is resumable and the cheap part can be redone:

    uv run c4-build-data --out runs/c4data-v2 --shards 500 --workers 2   # solve games into shards
    uv run c4-assemble --shards runs/c4data-v2 --out data/c4v2             # label, split, eval sets

Positions come from games played with the solver: at each ply the side to move
plays a random optimal move with probability 1 - eps, otherwise a random move
that neither wins at once (so the game goes on) nor lets the opponent win at once
(when it can avoid that). eps = 1 is safe-random play. Every position from ply
MIN_PLY to MAX_PLY along a game is kept with the solver's score for every column,
so each solve yields a dataset position.

Splits are by game, and any position whose canonical form (the smaller of the
board and its mirror image) occurs in the test games is removed from training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .connect4 import INVALID, ROOT, W, Board, Solver, board_from_moves, depth_from_score, label

MIN_PLY, MAX_PLY = 6, 36
EPS_MIX = ((0.1, 0.25), (0.3, 0.25), (1.0, 0.5))  # (eps, share of games)
STAGES = ((6, 16), (17, 26), (27, 36))           # game stage by plies played
R_BUCKETS = ((1, 1), (2, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 16), (17, 42))
D_BUCKETS = ((1, 1), (2, 2), (3, 4), (5, 6), (7, 8), (9, 12), (13, 16), (17, 24), (25, 42))


# ----------------------------------------------------------------- generation


def play_game(solver: Solver, rng: np.random.Generator, eps: float) -> list[tuple[str, np.ndarray]]:
    """One game; returns (move string, per-column scores) for plies MIN_PLY..MAX_PLY."""
    b, out = Board(), []
    while True:
        n, ms = b.n, b.move_string()
        sc = solver.analyze([ms])[0]
        if n >= MIN_PLY:
            out.append((ms, sc.copy()))
        if n >= MAX_PLY:
            return out
        legal = [c for c in range(W) if sc[c] != INVALID]
        if rng.random() >= eps:
            best = max(sc[c] for c in legal)
            c = int(rng.choice([c for c in legal if sc[c] == best]))
        else:
            no_win = [c for c in legal if not (sc[c] > 0 and depth_from_score(int(sc[c]), n) == 1)]
            safe = [c for c in no_win if not (sc[c] < 0 and depth_from_score(int(sc[c]), n) == 2)]
            c = int(rng.choice(safe or no_win or legal))
        if b.is_winning_move(c):
            return out
        b.play(c)
        if not b.legal():
            return out


def build_shard(args) -> str:
    """Solve games until `positions` positions; write <out>/shard_<id>.npz atomically."""
    shard, positions, out_dir, seed = args
    path = Path(out_dir) / f"shard_{shard:04d}.npz"
    if path.exists():
        return f"{path.name} exists"
    rng = np.random.default_rng([seed, shard])
    eps_values, eps_p = zip(*EPS_MIX)
    solver = Solver()
    t0, moves, scores, games, eps_of = time.time(), [], [], [], []
    g = 0
    while len(moves) < positions:
        eps = float(rng.choice(eps_values, p=eps_p))
        for ms, sc in play_game(solver, rng, eps):
            moves.append(ms)
            scores.append(sc)
            games.append(shard * 1_000_000 + g)
            eps_of.append(eps)
        g += 1
    solver.close()
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, moves=np.array(moves), scores=np.array(scores, dtype=np.int16),
                        game=np.array(games, dtype=np.int64), eps=np.array(eps_of, dtype=np.float32))
    tmp.replace(path)
    return f"{path.name}: {len(moves)} positions from {g} games in {time.time() - t0:.0f}s"


def build_main(argv=None):
    p = argparse.ArgumentParser(description="Solve games into dataset shards (resumable: existing shards are skipped)")
    p.add_argument("--out", default="runs/c4data-v2")
    p.add_argument("--shards", type=int, default=500)
    p.add_argument("--positions-per-shard", type=int, default=5000)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for tmp in out.glob("*.tmp.npz"):  # partial shard from an interrupted run
        tmp.unlink()
    todo = [s for s in range(args.shards) if not (out / f"shard_{s:04d}.npz").exists()]
    print(f"{len(todo)} of {args.shards} shards to build with {args.workers} workers", flush=True)
    jobs = [(s, args.positions_per_shard, str(out), args.seed) for s in todo]
    with ProcessPoolExecutor(args.workers) as ex:
        for i, msg in enumerate(ex.map(build_shard, jobs), 1):
            print(f"{time.strftime('%H:%M:%S')} [{i}/{len(todo)}] {msg}", flush=True)
    print("done", flush=True)


# ------------------------------------------------------------------- assembly


def canonical_keys(tokens: np.ndarray) -> np.ndarray:
    """Stable 64-bit hash of the smaller of each position and its mirror image, from the
    (N, 42) side-to-move tokens. Equal boards have equal piece counts, hence the same
    side to move, so tokens identify boards exactly."""
    t = np.ascontiguousarray(tokens.reshape(-1, 6, 7))
    m = np.ascontiguousarray(t[:, :, ::-1])
    keys = np.empty(len(t), dtype=np.int64)
    for i in range(len(t)):
        a, b = t[i].tobytes(), m[i].tobytes()
        keys[i] = int.from_bytes(hashlib.blake2b(min(a, b), digest_size=8).digest(), "little", signed=True)
    return keys


def _subset(d: dict, idx: np.ndarray) -> dict:
    n = len(d["moves"])
    return {k: v[idx] if isinstance(v, np.ndarray) and v.ndim and len(v) == n else v for k, v in d.items()}


def stage_of(n_moves: np.ndarray) -> np.ndarray:
    s = np.full(len(n_moves), -1)
    for i, (lo, hi) in enumerate(STAGES):
        s[(n_moves >= lo) & (n_moves <= hi)] = i
    return s


def bucket(x: np.ndarray, buckets) -> np.ndarray:
    b = np.full(len(x), -1)
    for i, (lo, hi) in enumerate(buckets):
        b[(x >= lo) & (x <= hi)] = i
    return b


def assemble_main(argv=None):
    p = argparse.ArgumentParser(description="Label shards, split by game, build train / tactical / general sets")
    p.add_argument("--shards", default="runs/c4data-v2")
    p.add_argument("--out", default=str(ROOT / "data" / "c4v2"))
    p.add_argument("--test-frac", type=float, default=0.15, help="fraction of games held out for evaluation")
    p.add_argument("--tactical-per-cell", type=int, default=500, help="per (r bucket, game stage) cell")
    p.add_argument("--general-per-bucket", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    # Label one shard at a time: label() builds a Python Board per position, so labelling
    # everything at once would need several GB on a machine that also serves DNS.
    files = sorted(Path(args.shards).glob("shard_*[0-9].npz"))
    chunks = []
    for i, f in enumerate(files):
        with np.load(f) as z:
            part = label(list(z["moves"]), z["scores"])
            part.update(game=z["game"], eps=z["eps"])
        part["canon"] = canonical_keys(part["tokens"])
        chunks.append(part)
        if (i + 1) % 50 == 0:
            print(f"labelled {i + 1}/{len(files)} shards", flush=True)
    d = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    del chunks
    moves, game = d["moves"], d["game"]
    print(f"{len(files)} shards, {len(moves)} positions, {len(np.unique(game))} games", flush=True)

    # Split by game; dedupe within each split; drop train positions whose canonical form is in test.
    rng = np.random.default_rng(args.seed)
    games = np.unique(game)
    test_games = set(rng.choice(games, int(len(games) * args.test_frac), replace=False).tolist())
    is_test = np.array([g in test_games for g in game])
    test_idx = np.flatnonzero(is_test)
    test_idx = test_idx[np.unique(d["canon"][test_idx], return_index=True)[1]]
    test_keys = set(d["canon"][test_idx].tolist())
    train_idx = np.flatnonzero(~is_test)
    train_idx = train_idx[np.unique(d["canon"][train_idx], return_index=True)[1]]
    leaked = np.array([k in test_keys for k in d["canon"][train_idx]])
    train_idx = np.sort(train_idx[~leaked])
    test_idx = np.sort(test_idx)
    assert not set(d["canon"][train_idx].tolist()) & test_keys, "train/test overlap"

    test = _subset(d, test_idx)
    stage = stage_of(test["n_moves"])
    unique_win = (test["value"] == 1) & (test["optimal"].sum(1) == 1)
    rb = bucket(test["req_depth"], R_BUCKETS)

    # Tactical set: unique winning move, balanced over (r bucket, game stage) cells.
    tac, cells = [], {}
    for i, (lo, hi) in enumerate(R_BUCKETS):
        for j, (slo, shi) in enumerate(STAGES):
            pool = np.flatnonzero(unique_win & (rb == i) & (stage == j))
            take = rng.permutation(pool)[:args.tactical_per_cell]
            tac.append(take)
            cells[f"r{lo}-{hi}|ply{slo}-{shi}"] = dict(available=int(len(pool)), taken=int(len(take)))
    tac = np.sort(np.concatenate(tac))

    # General set: old protocol, balanced by d bucket (decided) plus draws.
    db = bucket(test["depth"], D_BUCKETS)
    db[test["value"] == 0] = len(D_BUCKETS)
    gen = np.sort(np.concatenate([rng.permutation(np.flatnonzero(db == k))[:args.general_per_bucket]
                                  for k in range(len(D_BUCKETS) + 1)]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keep = ("tokens", "moves", "n_moves", "scores", "legal", "optimal", "value", "score", "depth",
            "req_depth", "game", "eps", "canon")
    for name, idx_or_d in (("train", _subset(d, train_idx)), ("tactical", _subset(test, tac)),
                           ("general", _subset(test, gen))):
        np.savez_compressed(f"{out}_{name}.npz", **{k: idx_or_d[k] for k in keep})
    meta = dict(shards=len(files), positions=int(len(moves)), games=int(len(games)),
                test_games=len(test_games), train=int(len(train_idx)), test_pool=int(len(test_idx)),
                train_removed_as_test_duplicates=int(leaked.sum()), tactical=int(len(tac)),
                general=int(len(gen)), tactical_cells=cells, eps_mix=EPS_MIX, plies=[MIN_PLY, MAX_PLY])
    Path(f"{out}_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "tactical_cells"}, indent=2))
    print("tactical cells (available / taken):")
    for k, v in cells.items():
        print(f"  {k:>18}: {v['available']:6d} / {v['taken']}")


# ------------------------------------------------------------------ baselines


def _wins_at(b: Board, c: int, player: int) -> bool:
    """Would `player` connect four by dropping a piece in column c?"""
    r, g = b.heights[c], b.grid
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for sgn in (1, -1):
            rr, cc = r + sgn * dr, c + sgn * dc
            while 0 <= rr < 6 and 0 <= cc < W and g[rr, cc] == player:
                count += 1
                rr, cc = rr + sgn * dr, cc + sgn * dc
        if count >= 4:
            return True
    return False


def shortcut_accuracy(moves: str, optimal: np.ndarray) -> float:
    """Expected accuracy of 'win now, else block, else a random move that doesn't
    hand the opponent a win, else any random move'."""
    from .connect4 import opponent_can_win_after
    b = board_from_moves(moves)
    legal, me = b.legal(), b.to_move
    choice = ([c for c in legal if _wins_at(b, c, me)]
              or [c for c in legal if _wins_at(b, c, 3 - me)]
              or [c for c in legal if not opponent_can_win_after(b, c)]
              or legal)
    return sum(bool(optimal[c]) for c in choice) / len(choice)


def baselines_main(argv=None):
    """Stage 2: search and shortcut baselines on the tactical set, by r bucket and game stage."""
    from .connect4 import search_baseline_accuracy
    p = argparse.ArgumentParser(description=baselines_main.__doc__)
    p.add_argument("--data", default=str(ROOT / "data" / "c4v2_tactical.npz"))
    p.add_argument("--out", default="results/c4v2-baselines")
    p.add_argument("--max-k", type=int, default=20)
    args = p.parse_args(argv)
    with np.load(args.data) as f:
        d = {k: f[k] for k in f.files}
    n, r, sc, opt, legal = d["n_moves"], d["req_depth"], d["scores"], d["optimal"], d["legal"]
    ks = list(range(1, args.max_k + 1))
    search = np.array([[search_baseline_accuracy(s, int(m), k) for k in ks] for s, m in zip(sc, n)])
    shortcut = np.array([shortcut_accuracy(str(m), o) for m, o in zip(d["moves"], opt)])
    chance = opt.sum(1) / legal.sum(1)
    rb, st = bucket(r, R_BUCKETS), stage_of(n)
    rows = []
    for i, (lo, hi) in enumerate(R_BUCKETS):
        for j in range(-1, len(STAGES)):
            m = (rb == i) & ((st == j) if j >= 0 else True)
            if not m.any():
                continue
            rows.append(dict(r=f"{lo}-{hi}", stage="all" if j < 0 else f"{STAGES[j][0]}-{STAGES[j][1]}",
                             n=int(m.sum()), chance=round(float(chance[m].mean()), 4),
                             shortcut=round(float(shortcut[m].mean()), 4),
                             search_by_k=[round(float(v), 4) for v in search[m].mean(0)]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "baselines.json").write_text(json.dumps(dict(data=args.data, ks=ks, rows=rows), indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    allrows = [x for x in rows if x["stage"] == "all"]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(allrows)))
    for x, c in zip(allrows, colors):
        ax.plot(ks, x["search_by_k"], marker="o", ms=2.5, color=c, label=f"r={x['r']} (n={x['n']})")
    ax.set_xlabel("search depth k (plies), no evaluation function")
    ax.set_ylabel("optimal-move accuracy")
    ax.set_title("Reference: depth-limited search on the tactical set, by required depth r")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "search_by_depth.png", dpi=150)
    print(f"{'r':>6} {'n':>5} {'chance':>7} {'shortcut':>9}  search k=1,2,4,8,12,16")
    for x in allrows:
        s = x["search_by_k"]
        print(f"{x['r']:>6} {x['n']:5d} {x['chance']:7.3f} {x['shortcut']:9.3f}  "
              + " ".join(f"{s[k - 1]:.2f}" for k in (1, 2, 4, 8, 12, 16)))
