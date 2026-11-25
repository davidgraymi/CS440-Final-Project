"""Summarise results/c4p/tournament/shard-*/games.jsonl into one JSON for the results page.
Run from the repo root: python scripts/c4p/tournament_analysis.py results/c4p/reports/tournament.json"""
import glob, json, sys, collections
import numpy as np

ROOT = "results/c4p/tournament"
G = [json.loads(l) for f in sorted(glob.glob(f"{ROOT}/shard-*/games.jsonl")) for l in open(f)]
VAL = {}
for f in glob.glob(f"{ROOT}/shard-*/openings.json"):
    VAL.update(json.load(open(f)))
P = ["perfect", "D0", "D1", "C", "A", "k24", "k8", "k4", "k1"]
ix = {p: i for i, p in enumerate(P)}
n = len(P)
rng = np.random.default_rng(0)

# ---- score matrix
S = np.zeros((n, n)); C = np.zeros((n, n)); W = np.zeros((n, n)); D = np.zeros((n, n))
for g in G:
    a, b, s = ix[g["a"]], ix[g["b"]], g["score_a"]
    S[a, b] += s; S[b, a] += 1 - s; C[a, b] += 1; C[b, a] += 1
    W[a, b] += s == 1; W[b, a] += s == 0; D[a, b] += s == 0.5; D[b, a] += s == 0.5


def bt(S, C, iters=3000):
    """Bradley-Terry by MM (draws count half). Returns Elo with mean 1500."""
    w = S.sum(1); g = np.ones(n)
    for _ in range(iters):
        den = (C / (g[:, None] + g[None, :] + 1e-12)).sum(1)
        g = np.maximum(w, 1e-9) / den
        g /= np.exp(np.log(g).mean())
    e = 400 * np.log10(g)
    return e - e.mean() + 1500


elo = bt(S, C)
# bootstrap over openings (each opening's games resampled together)
by_open = collections.defaultdict(list)
for g in G:
    by_open[g["opening"]].append(g)
opens = list(by_open)
per_open = {}
for o in opens:
    s = np.zeros((n, n)); c = np.zeros((n, n))
    for g in by_open[o]:
        a, b, x = ix[g["a"]], ix[g["b"]], g["score_a"]
        s[a, b] += x; s[b, a] += 1 - x; c[a, b] += 1; c[b, a] += 1
    per_open[o] = (s, c)
SO = np.stack([per_open[o][0] for o in opens]); CO = np.stack([per_open[o][1] for o in opens])
boots = []
for _ in range(300):
    pick = rng.integers(0, len(opens), len(opens))
    boots.append(bt(SO[pick].sum(0), CO[pick].sum(0), 600))
boots = np.array(boots)
elo_ci = np.percentile(boots, [2.5, 97.5], 0)

# ---- per depth of opening
depth_score = {p: {} for p in P}
for d in range(5):
    for p in P:
        s = c = 0
        for g in G:
            if len(g["opening"]) != d:
                continue
            if g["a"] == p: s += g["score_a"]; c += 1
            elif g["b"] == p: s += 1 - g["score_a"]; c += 1
        depth_score[p][d] = [round(s / c, 4) if c else None, c]

# ---- versus perfect: by theoretical value of the opening for the model's side
vs_perf = {}
for p in P[1:]:
    rows = {"win": [0, 0, 0, 0], "draw": [0, 0, 0, 0], "loss": [0, 0, 0, 0]}  # W D L n
    for g in G:
        if {g["a"], g["b"]} != {p, "perfect"}:
            continue
        v = VAL[g["opening"]]  # for the side to move after the opening
        v = v if g["first"] == p else -v
        kind = "win" if v > 0 else "loss" if v < 0 else "draw"
        s = g["score_a"] if g["a"] == p else 1 - g["score_a"]
        r = rows[kind]
        r[0] += s == 1; r[1] += s == 0.5; r[2] += s == 0; r[3] += 1
    vs_perf[p] = rows
open_vals = collections.Counter(("win" if v > 0 else "loss" if v < 0 else "draw") for v in VAL.values())

# ---- steps per move in games (adaptive models)
steps = {}
for p in ["D0", "D1", "C", "A"]:
    by_ply = collections.defaultdict(list)
    all_s = []
    by_opp = collections.defaultdict(list)
    for g in G:
        for side, key in (("a", "steps_a"), ("b", "steps_b")):
            if g[side] != p:
                continue
            opp = g["b"] if side == "a" else g["a"]
            o = len(g["opening"])
            mine_first = g["first"] == p
            for j, s in enumerate(g[key]):
                ply = o + 2 * j + (0 if mine_first else 1)  # pieces on board before the move
                by_ply[ply].append(s); all_s.append(s); by_opp[opp].append(s)
    steps[p] = dict(mean=round(float(np.mean(all_s)), 3),
                    hist=[round(float(x), 4) for x in np.bincount(all_s, minlength=25)[1:] / len(all_s)],
                    by_ply={k: [round(float(np.mean(v)), 3), len(v)] for k, v in sorted(by_ply.items()) if len(v) >= 50},
                    by_opp={k: round(float(np.mean(v)), 3) for k, v in by_opp.items()})

# steps in won vs lost games, and steps in the moves just before a loss
lens = {p: round(float(np.mean([g["plies"] for g in G if p in (g["a"], g["b"])])), 2) for p in P}
first_adv = dict(first=round(float(np.mean([g["score_a"] if g["first"] == g["a"] else 1 - g["score_a"] for g in G])), 4))
out = dict(players=P, n_games=len(G), n_openings=len(opens), open_vals=open_vals,
           games_per_pair=int(C[0, 1]) if C[0, 1] else None,
           score=(S / np.maximum(C, 1)).round(4).tolist(), count=C.astype(int).tolist(),
           wins=W.astype(int).tolist(), draws=D.astype(int).tolist(),
           elo=elo.round(1).tolist(), elo_lo=elo_ci[0].round(1).tolist(), elo_hi=elo_ci[1].round(1).tolist(),
           overall=[round(float(S[i].sum() / C[i].sum()), 4) for i in range(n)],
           depth_score=depth_score, depth_n={d: sum(1 for o in opens if len(o) == d) for d in range(5)},
           vs_perfect=vs_perf, steps=steps, length=lens, first_mover=first_adv)
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps({k: out[k] for k in ("n_games", "n_openings", "elo", "elo_lo", "elo_hi", "overall", "open_vals", "first_mover")}))
for p in P[1:]:
    print(p, vs_perf[p])
for p in steps:
    print(p, steps[p]["mean"], list(steps[p]["by_ply"].items())[::6])
