"""Split-half nested check of the blend weights, on FOLD-RANKED OOFs: fit the coordinate-search weights on one stratified half of the OOF rows, score the
other half, both directions — same as nested_check.py except each OOF is ranked inside its own _kK_sS fold (assemble_fr.py's foldrank)
instead of globally, so cross-fold calibration noise never enters the weight fit.
Usage: python nested_check_fr.py"""
import numpy as np, pandas as pd, os, sys, re
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata
from scipy.optimize import minimize
r = lambda a: rankdata(a) / len(a)
y = (pd.read_csv("data/train.csv", usecols=["Will_Buy_EV"]).Will_Buy_EV == "Yes").astype(int).values
def foldrank(name, v):
    """Rank v inside each StratifiedKFold fold parsed from the name's _k<K>_s<S>; names without that pattern
    (pub_*, old fixed-seed runs) fall back to the global rank r."""
    m = re.search(r'_k(\d+)_s(\d+)', name)
    if not m: return r(v)
    out = np.zeros(len(v))
    for a, b in StratifiedKFold(int(m.group(1)), shuffle=True, random_state=int(m.group(2))).split(v, y): out[b] = rankdata(v[b]) / len(b)
    return out
G = {"hybrid10": ["v27_hybrid_k10_s%s" % s for s in (42, 7, 2026, 101, 202, 303, 404, 505)],
     "pub_naji_01blend": ["pub_naji_01blend"], "pub_naji_xgb": ["pub_naji_xgb"], "pub_mega_B": ["pub_mega_B"], "pub_mega_D": ["pub_mega_D"],
     "viewD": ["v33_viewD_k10_s%s" % s for s in (42, 7, 2026, 101, 202)], "viewC_xgb": ["v33_viewC_xgb_k10_s42"], "viewD_init50": ["v33_viewD_initinc50_k10_s42"],
     "init": ["v34_init_inc_exact_k10_s%s" % s for s in (42, 7, 2026, 101, 202)], "init100": ["v34_init_inc100_k10_s%s" % s for s in (42, 7, 2026, 101, 202)],
     "realmlp_own10": ["v35_realmlp_k10_s%s" % s for s in (42, 7, 2026, 101, 202)]}
import ast
_g = next(n for n in ast.parse(open("assemble.py").read()).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "GROUPS")
for k, (ns, _) in eval(compile(ast.fix_missing_locations(ast.Expression(_g.value)), "assemble.py", "eval")).items():
    ns = [n for n in ns if os.path.exists(f"submissions/oof_{n}.npy")]
    if ns and k not in G: G[k] = ns   # every other assemble.py group, by its own seeds (2026-09-27)
G["init100"] = ["v34_init_inc100_k10_s%s" % s for s in (42, 7, 2026, 101, 202, 303, 404, 505)]   # v23-era 8-seed group
O = {g: np.mean([foldrank(n, np.load(f"submissions/oof_{n}.npy")) for n in ns], 0) for g, ns in G.items()}
if os.environ.get("PROBIT") == "1":   # blend in probit space: ndtri of each group's fold-ranked average (issue #1)
    from scipy.special import ndtri; O = {g: ndtri(np.clip(v, 1e-6, 1 - 1e-6)) for g, v in O.items()}
SETS = {"v18 groups": ["hybrid10", "pub_naji_01blend", "pub_naji_xgb", "pub_mega_B", "pub_mega_D", "viewD", "viewC_xgb", "init", "realmlp_own10"],
        "v20 groups": ["hybrid10", "pub_naji_01blend", "pub_naji_xgb", "pub_mega_B", "pub_mega_D", "viewD", "viewC_xgb", "viewD_init50", "init", "init100", "realmlp_own10"]}
V23 = ["pub_naji_01blend", "pub_naji_xgb", "viewD_init50", "viewC_init100", "viewD_init500", "init", "realmlp_own10", "init100"]
V26 = ["pub_naji_01blend", "viewD", "viewD_init50", "init", "realmlp_own10", "init100_k20", "pub_legtarrr_v19", "agg_init100", "init100", "init100_slow", "init100pkm" if False else "init_combo"]
V26N = ["pub_naji_01blend", "viewD", "viewD_init50", "init", "realmlp_own10", "init100_k20", "agg_init100", "init100"]
V27N = ["pub_naji_01blend", "viewD", "viewD_init50", "init", "realmlp_own10", "init100_k20", "agg_init100", "agg_init100_k20"]
_SETS_09_27b = {"v26fr nov19 groups (submitted)": V26N, "v27fr nov19 groups": V27N}   # 2026-09-27 evening
_SETS_09_28 = {"v27fr nov19 groups (submitted)": V27N, "v27 + hj_xgb 8-seed bag": V27N + ["hj_xgb"]}   # 2026-09-28, issue #1
_SETS_09_29a = {"v30 (hj_xgb, submitted)": V27N + ["hj_xgb"], "v30 + hj_k20": V27N + ["hj_xgb", "hj_k20"], "v27 + hj_k20 only": V27N + ["hj_k20"]}   # 2026-09-29, issue #1
V31 = V27N + ["hj_k20"]
_SETS_09_29b = {"v31 groups (submitted)": V31, "v31 + realmlp_k20": V31 + ["realmlp_k20"], "v31, realmlp_k20 for own10": [g for g in V31 if g != "realmlp_own10"] + ["realmlp_k20"]}   # 2026-09-29 04:10, issue #1
V33 = [g for g in V31 if g != "realmlp_own10"] + ["realmlp_k20"]
_SETS_09_29c = {"v33 groups (submitted)": V33, "v33 + pub_hjlr2": V33 + ["pub_hjlr2"], "v33 + pub_hjlr2 + pub_hjlr": V33 + ["pub_hjlr2", "pub_hjlr"]}   # 2026-09-29 08:45, issue #1
V34 = V33 + ["pub_hjlr2", "pub_hjlr"]
_SETS_09_29d = {"v34 groups (submitted)": V34, "v34 + init100_tok": V34 + ["init100_tok"]}   # 2026-09-29 10:15, issue #1
_SETS_09_29e = {"v34 groups (submitted)": V34, "v34 + init100_tok_k20": V34 + ["init100_tok_k20"]}   # 2026-09-29 evening (1 seed), issue #1
SETS = {"v34 groups": V34, "v35 = v34 + init100_tok_k20 (4 seeds)": V34 + ["init100_tok_k20"]}   # 2026-09-29 late, 4 seeds, issue #1
_SETS_09_27 = {"v23 groups": V23, "v26fr groups": V26, "v26fr groups minus legtarrr v19": [g for g in V26 if g != "pub_legtarrr_v19"]}   # 2026-09-27
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
