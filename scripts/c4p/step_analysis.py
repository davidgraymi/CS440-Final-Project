"""Outer-step-length analysis of the main-phase models. Reads results/c4p/*/eval-*.npz (same positions
in every file), writes a JSON summary. Run from the repo root with the project venv:
    python scripts/c4p/step_analysis.py results/c4p/reports/outer-steps.json"""
import json, sys
import numpy as np

RUNS = {  # key: (dir, label, kind)
    "D0": ("c4p-long-D", "Adaptive D · seed 0", "adaptive"),
    "D1": ("c4p-main-s1", "Adaptive D · seed 1", "adaptive"),
    "C": ("c4p-long-C", "Adaptive C · full credit only", "adaptive"),
    "A": ("c4p-long-A", "Adaptive A · no credit window", "adaptive"),
    "k24": ("c4p-fixed-k24", "Fixed 24 steps", "fixed"),
    "k8": ("c4p-fixed-k8", "Fixed 8 steps", "fixed"),
    "k4": ("c4p-fixed-k4", "Fixed 4 steps", "fixed"),
    "k1": ("c4p-fixed-k1", "Fixed 1 step", "fixed"),
}
BUCKETS = [("1", 1, 1), ("2", 2, 2), ("3–4", 3, 4), ("5–6", 5, 6), ("7–8", 7, 8), ("9–10", 9, 10),
           ("11–12", 11, 12), ("13–16", 13, 16), ("17+", 17, 99)]
SHALLOW, DEEP = (1, 4), (9, 99)
THRESH = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
NB = 2000
rng = np.random.default_rng(0)


def stop_at(p, t):
    return (np.cumsum(p, 1) >= t).argmax(1) + 1


def r3(x):
    return round(float(x), 4)


def boot_diff(a, b, idx_sets):
    """paired bootstrap of mean(a) - mean(b) over positions; returns diff, 95% CI, two-sided p."""
    d = a.astype(np.float32) - b.astype(np.float32)
    bs = d[idx_sets].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return dict(diff=r3(d.mean()), lo=r3(lo), hi=r3(hi), p=max(float(p), 1 / NB))


out = {}
for split in ("tactical", "general"):
    Z = {k: np.load(f"results/c4p/{d}/eval-{split}.npz") for k, (d, _, _) in RUNS.items()}
    r = Z["D0"]["req_depth"].astype(int)
    n = len(r)
    idx_sets = rng.integers(0, n, size=(NB, n))
    S = {"n": n, "models": {}, "pairs": {}, "buckets": []}
    masks = {}
    for name, lo, hi in BUCKETS:
        m = (r >= lo) & (r <= hi)
        if m.sum():
            masks[name] = m
            S["buckets"].append(dict(name=name, n=int(m.sum())))
    if (r <= 0).sum():
        masks["no forced win"] = r <= 0
        S["buckets"].append(dict(name="no forced win", n=int((r <= 0).sum())))
    sh, dp = (r >= SHALLOW[0]) & (r <= SHALLOW[1]), (r >= DEEP[0])
    at_stop = {}
    for k, (d, label, kind) in RUNS.items():
        ok, p = Z[k]["move_ok"], Z[k]["p"]
        K = ok.shape[1]
        stop = stop_at(p, 0.5) if kind == "adaptive" else np.full(n, K)
        a = ok[np.arange(n), stop - 1]
        at_stop[k] = a
        M = dict(label=label, kind=kind, K=K, acc=r3(a.mean()), mean_stop=r3(stop.mean()),
                 curve=[r3(x) for x in ok.mean(0)],                      # forced to stop at step k
                 curve_shallow=[r3(x) for x in ok[sh].mean(0)], curve_deep=[r3(x) for x in ok[dp].mean(0)],
                 by_bucket={b: dict(acc=r3(a[m].mean()), stop=r3(stop[m].mean()),
                                    curve=[r3(x) for x in ok[m].mean(0)]) for b, m in masks.items()})
        bs = a[idx_sets].mean(1)
        M["ci"] = [r3(np.percentile(bs, 2.5)), r3(np.percentile(bs, 97.5))]
        if kind == "adaptive":
            # accuracy/compute trade-off from the stop threshold (0.5 = the model's own rule)
            M["frontier"] = []
            for t in THRESH:
                s = stop_at(p, t)
                M["frontier"].append(dict(t=t, steps=r3(s.mean()), acc=r3(ok[np.arange(n), s - 1].mean())))
            # same model, same average budget, but every position gets the same step count
            kk = int(round(stop.mean()))
            M["own_fixed"] = dict(k=kk, acc=r3(ok[:, kk - 1].mean()),
                                  vs=boot_diff(a, ok[:, kk - 1], idx_sets))
            M["best_fixed_k"] = int(ok.mean(0).argmax() + 1)
            M["best_fixed_acc"] = r3(ok.mean(0).max())
            M["oracle"] = r3(ok.any(1).mean())   # some step gets it right
            last = ok[:, -1]
            M["after_stop"] = dict(fixed_later=r3((~a & last).mean()), broke_later=r3((a & ~last).mean()),
                                   acc_k24=r3(last.mean()))
            # how often accuracy goes down after first being right (overthinking)
            first = np.where(ok.any(1), ok.argmax(1), K)
            M["flip_after_right"] = r3(((~ok) & (np.arange(K)[None] > first[:, None])).any(1).mean())
            hist = {}
            for b, m in masks.items():
                h = np.bincount(stop[m], minlength=K + 1)[1:]
                hist[b] = [r3(x) for x in h / m.sum()]
            M["stop_hist"] = hist
            # first step at which each position is solved and stays solved
            stays = np.flip(np.cumprod(np.flip(ok, 1), 1), 1).astype(bool)
            settle = np.where(stays.any(1), stays.argmax(1) + 1, K + 1)
            M["settle_by_bucket"] = {b: r3(np.median(settle[m])) for b, m in masks.items()}
            M["stop_minus_settle"] = {b: r3(np.mean(stop[m] - np.minimum(settle[m], K))) for b, m in masks.items()}
        S["models"][k] = M
    for a_ in ("D0", "D1", "C"):
        for b_ in ("k24", "k8", "k4", "k1", "C", "A"):
            if a_ == b_:
                continue
            row = dict(all=boot_diff(at_stop[a_], at_stop[b_], idx_sets))
            for nm_, m in (("deep", dp), ("shallow", sh)):
                ii = np.flatnonzero(m)
                row[nm_] = boot_diff(at_stop[a_][ii], at_stop[b_][ii], rng.integers(0, len(ii), size=(NB, len(ii))))
            S["pairs"][f"{a_}-{b_}"] = row
    out[split] = S
json.dump(out, open(sys.argv[1], "w"), indent=1)
print("ok")
