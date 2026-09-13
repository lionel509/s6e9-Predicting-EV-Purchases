"""Split-half nested check of the blend weights: fit the coordinate-search weights on one stratified half of the OOF rows, score the
other half, both directions. Answers whether a higher in-sample blend OOF (v20 0.946495 vs v18 0.946479) survives out of sample,
after the public board moved the other way (0.94644 vs 0.94646). Usage: python nested_check.py"""
import numpy as np, pandas as pd, os, sys
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata
from scipy.optimize import minimize
r = lambda a: rankdata(a) / len(a)
y = (pd.read_csv("data/train.csv", usecols=["Will_Buy_EV"]).Will_Buy_EV == "Yes").astype(int).values
G = {"hybrid10": ["v27_hybrid_k10_s%s" % s for s in (42, 7, 2026, 101, 202, 303, 404, 505)],
     "pub_naji_01blend": ["pub_naji_01blend"], "pub_naji_xgb": ["pub_naji_xgb"], "pub_mega_B": ["pub_mega_B"], "pub_mega_D": ["pub_mega_D"],
     "viewD": ["v33_viewD_k10_s%s" % s for s in (42, 7, 2026, 101, 202)], "viewC_xgb": ["v33_viewC_xgb_k10_s42"], "viewD_init50": ["v33_viewD_initinc50_k10_s42"],
     "init": ["v34_init_inc_exact_k10_s%s" % s for s in (42, 7, 2026, 101, 202)], "init100": ["v34_init_inc100_k10_s%s" % s for s in (42, 7, 2026, 101, 202)],
     "realmlp_own10": ["v35_realmlp_k10_s%s" % s for s in (42, 7, 2026, 101, 202)]}
O = {g: np.mean([r(np.load(f"submissions/oof_{n}.npy")) for n in ns], 0) for g, ns in G.items()}
SETS = {"v18 groups": ["hybrid10", "pub_naji_01blend", "pub_naji_xgb", "pub_mega_B", "pub_mega_D", "viewD", "viewC_xgb", "init", "realmlp_own10"],
        "v20 groups": ["hybrid10", "pub_naji_01blend", "pub_naji_xgb", "pub_mega_B", "pub_mega_D", "viewD", "viewC_xgb", "viewD_init50", "init", "init100", "realmlp_own10"]}
def fit_weights(M, yy):
    n = M.shape[1]; auc = lambda w: roc_auc_score(yy, M @ (w / w.sum())); w = np.ones(n) / n; best = auc(w)
    grid = [0, .01, .02, .03, .05, .07, .1, .13, .17, .22, .28, .35, .45, .6]
    for p in range(3):
        improved = False
        for j in range(n):
            for g in grid:
                w2 = w.copy(); w2[j] = g
                if w2.sum() == 0: continue
                a = auc(w2)
                if a > best + 1e-7: best, w, improved = a, w2, True
        if not improved: break
    return w / w.sum(), best
halves = list(StratifiedKFold(2, shuffle=True, random_state=7).split(y, y))
for name, gs in SETS.items():
    M = np.column_stack([O[g] for g in gs]); full_w, full = fit_weights(M, y); outs = []
    for a, b in halves:
        w, _ = fit_weights(M[a], y[a]); outs.append(roc_auc_score(y[b], M[b] @ w))
    print(f"{name:11s} in-sample {full:.6f}   nested halves {outs[0]:.6f} / {outs[1]:.6f}  mean {np.mean(outs):.6f}", flush=True)
    print("   weights:", {g: round(float(x), 3) for g, x in zip(gs, full_w) if x > 0}, flush=True)
