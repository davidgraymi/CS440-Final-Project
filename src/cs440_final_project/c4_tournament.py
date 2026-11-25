"""Round-robin Connect Four tournament between c4-ponder checkpoints (and a perfect solver player).

Every model is deterministic (argmax move), so one game per pairing from the empty board would
just repeat. Instead each pair plays from a suite of openings: every distinct position reachable in
0..max_ply plies. From each opening the pair plays twice, once with each player to move first.

    python -m cs440_final_project.c4_tournament --max-ply 3 --out results/c4p/tournament

Writes games.jsonl (one line per game) and openings.json (opening -> solver value).
Adaptive models answer at their own stopping step (cumulative stop chance >= 1/2); fixed-step
controls answer at their last step. The step each move used is recorded."""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

from .c4_ponder import episode, halting_dist, stop_step
from .connect4 import H, W, Board, Solver, board_from_moves
from .models import ModelConfig, build_model
from .train import pick_device

DEFAULT_PLAYERS = {
    "D0": "runs/c4p-long-D/ckpt.pt", "D1": "runs/c4p-main-s1/ckpt.pt",
    "C": "runs/c4p-long-C/ckpt.pt", "A": "runs/c4p-long-A/ckpt.pt",
    "k24": "runs/c4p-fixed-k24/ckpt.pt", "k8": "runs/c4p-fixed-k8/ckpt.pt",
    "k4": "runs/c4p-fixed-k4/ckpt.pt", "k1": "runs/c4p-fixed-k1/ckpt.pt",
}
CENTER_ORDER = [3, 2, 4, 1, 5, 0, 6]


def ckpt_path(rel: str) -> str:
    """runs/<run>/ckpt.pt if present (a local training run), else the committed copy in results/c4p/."""
    p = Path(rel)
    return rel if p.exists() else str(Path("results/c4p") / p.parent.name / p.name)


class ModelPlayer:
    def __init__(self, ckpt: str, device, batch_size: int = 512):
        ck = torch.load(ckpt, map_location="cpu", weights_only=True)
        self.model = build_model("trm", ModelConfig(**ck["cfg"]))
        self.model.load_state_dict(ck["ema"])
        self.model.to(device).eval()
        self.fixed = ck.get("args", {}).get("fixed_steps", 0)
        self.k = self.fixed or ck["cfg"]["n_sup"]
        self.device, self.bs = device, batch_size

    @torch.no_grad()
    def moves(self, boards: list[Board]) -> tuple[list[int], list[int]]:
        cols, steps = [], []
        for i in range(0, len(boards), self.bs):
            x = torch.from_numpy(np.stack([b.tokens() for b in boards[i:i + self.bs]]).astype(np.int64)).to(self.device)
            outs = episode(self.model, x, self.k, ckpt=False)
            pol = torch.stack(outs["policy"], 1)  # (B, K, 7)
            if self.fixed:
                s = torch.full((len(x),), self.k, device=x.device, dtype=torch.long)
            else:
                s = stop_step(halting_dist(torch.stack(outs["q"], 1)))
            chosen = pol[torch.arange(len(x), device=x.device), s - 1].argmax(-1)
            cols += chosen.tolist()
            steps += s.tolist()
        return cols, steps


class PerfectPlayer:
    """Solver move; ties broken toward the centre column."""

    def __init__(self):
        self.solver, self.cache = Solver(), {}

    def scores(self, ms: str) -> np.ndarray:
        if ms not in self.cache:
            self.cache[ms] = self.solver.analyze([ms])[0]
        return self.cache[ms]

    def moves(self, boards: list[Board]) -> tuple[list[int], list[int]]:
        out = []
        for b in boards:
            sc = self.scores(b.move_string())
            best = max(sc[c] for c in b.legal())
            out.append(next(c for c in CENTER_ORDER if c in b.legal() and sc[c] == best))
        return out, [0] * len(boards)


def openings(max_ply: int) -> list[str]:
    """Distinct positions (by board) after 0..max_ply plies, as move strings (shortest first)."""
    seen, out = set(), []
    for n in range(max_ply + 1):
        for seq in itertools.product(range(W), repeat=n):
            b = Board()
            ok = True
            for c in seq:
                if b.heights[c] >= H or b.is_winning_move(c):
                    ok = False
                    break
                b.play(c)
            if ok and b.key() not in seen:
                seen.add(b.key())
                out.append(b.move_string())
    return out


def play_pair(pa, pb, names, opens: list[str]) -> list[dict]:
    """Every opening twice: first with pa to move, then with pb to move."""
    games = []
    for o in opens:
        for first, second in ((0, 1), (1, 0)):
            games.append(dict(board=board_from_moves(o), opening=o, first=first, steps=[[], []], result=None))
    players = (pa, pb)
    live = list(range(len(games)))
    while live:
        for who in (0, 1):
            idx = [i for i in live if games[i]["result"] is None and (games[i]["board"].n - len(games[i]["opening"])) % 2 == (0 if games[i]["first"] == who else 1)]
            if not idx:
                continue
            cols, steps = players[who].moves([games[i]["board"] for i in idx])
            for i, c, s in zip(idx, cols, steps):
                g, b = games[i], games[i]["board"]
                g["steps"][who].append(s)
                if c not in b.legal():  # policy is masked, so this should not happen
                    g["result"] = 1 - who
                elif b.is_winning_move(c):
                    b.play(c)
                    g["result"] = who
                else:
                    b.play(c)
                    if b.n == W * H:
                        g["result"] = 0.5
        live = [i for i in live if games[i]["result"] is None]
    out = []
    for g in games:
        r = g["result"]
        score_a = 0.5 if r == 0.5 else float(r == 0)
        out.append(dict(a=names[0], b=names[1], opening=g["opening"], first=names[g["first"]],
                        score_a=score_a, plies=g["board"].n, moves=g["board"].move_string(),
                        steps_a=g["steps"][0], steps_b=g["steps"][1]))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--max-ply", type=int, default=3)
    p.add_argument("--players", default=",".join(DEFAULT_PLAYERS) + ",perfect")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="results/c4p/tournament")
    p.add_argument("--limit-openings", type=int, default=0, help="dry runs")
    p.add_argument("--shard", type=int, default=0, help="play only pairs i, i+n, i+2n, ... (with --num-shards n)")
    p.add_argument("--num-shards", type=int, default=1)
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    names = args.players.split(",")
    opens = openings(args.max_ply)
    if args.limit_openings:
        opens = opens[:args.limit_openings]
    solver = Solver()
    val = {}
    for o in opens:
        sc = solver.analyze([o])[0]
        val[o] = int(max(s for s in sc if s > -1000))
    solver.close()
    (out / "openings.json").write_text(json.dumps(val))
    print(f"{len(opens)} openings up to {args.max_ply} plies; {len(names)} players", flush=True)
    used = {x for pr in list(itertools.combinations(names, 2))[args.shard::args.num_shards] for x in pr}
    players = {n: PerfectPlayer() if n == "perfect" else ModelPlayer(ckpt_path(DEFAULT_PLAYERS[n]), device) for n in used}
    done = set()
    gf = out / "games.jsonl"
    if gf.exists():  # resume: skip pairs already written
        for line in gf.open():
            g = json.loads(line)
            done.add((g["a"], g["b"]))
    with gf.open("a") as f:
        pairs = list(itertools.combinations(names, 2))[args.shard::args.num_shards]
        print("pairs:", pairs, flush=True)
        for a, b in pairs:
            if (a, b) in done:
                continue
            t = time.time()
            res = play_pair(players[a], players[b], (a, b), opens)
            for g in res:
                f.write(json.dumps(g) + "\n")
            f.flush()
            s = sum(g["score_a"] for g in res)
            print(f"{a} vs {b}: {s:.1f}/{len(res)}  ({time.time() - t:.0f} s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
