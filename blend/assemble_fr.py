"""Fold-ranked assembly: identical to assemble.py (same GROUPS, same weight fit, same test columns, same output)
except each OOF is ranked INSIDE its own fold before the group average — matching how test columns are rank-averaged
per fold model — instead of globally across folds, so cross-fold calibration noise no longer leaks into the weight
fit. GROUPS is parsed out of assemble.py's source (ast, the GROUPS assign) rather than copied, so the two can never
drift. Usage: python assemble_fr.py <out>   (edit GROUPS in assemble.py, not here)"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "models"))   # models/ holds features + the v* scripts
import sys, ast, re, numpy as np, pandas as pd
from scipy.stats import rankdata
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from features import load
out = sys.argv[1] if len(sys.argv) > 1 else "blend_v2"
_g = next(n for n in ast.parse(open("blend/assemble.py").read()).body
          if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "GROUPS")
GROUPS = eval(compile(ast.fix_missing_locations(ast.Expression(_g.value)), "blend/assemble.py", "eval"))
import os; GROUPS = {k: v for k, v in GROUPS.items() if k not in os.environ.get("EXCLUDE", "").split(",")}   # EXCLUDE=g1,g2 drops groups for a with/without comparison
tr, te, o, y, feats = load(); r = lambda a: rankdata(a) / len(a)
from sklearn.model_selection import StratifiedKFold
def foldrank(name, v):
    """Rank v inside each StratifiedKFold fold parsed from the name's _k<K>_s<S>; names without that pattern
    (pub_*, old fixed-seed runs) fall back to the global rank r."""
    m = re.search(r'_k(\d+)_s(\d+)', name)
    if not m: return r(v)
    out = np.zeros(len(v))
    for a, b in StratifiedKFold(int(m.group(1)), shuffle=True, random_state=int(m.group(2))).split(v, y): out[b] = rankdata(v[b]) / len(b)
    return out
import os
O, P, names = [], [], []
for g, (oofs, tests) in GROUPS.items():
    oofs = [n for n in oofs if os.path.exists(f"submissions/oof_{n}.npy")]
    if not oofs: print(f"  {g}: no OOF files yet, skipped"); continue
    oof = np.mean([foldrank(n, np.load(f"submissions/oof_{n}.npy")) for n in oofs], 0)
    if tests is None: tst = np.mean([r(pd.read_csv(f"submissions/{n}.csv")["Will_Buy_EV"].values) for n in oofs], 0)
    else:
        tests = [(n, w) for n, w in tests if os.path.exists(f"submissions/{n}.csv")]
        if not tests: tst = np.mean([r(pd.read_csv(f"submissions/{n}.csv")["Will_Buy_EV"].values) for n in oofs], 0); print(f"  {g}: refit missing, using fold-average test columns")
        else: tst = r(sum(w * pd.read_csv(f"submissions/{n}.csv")["Will_Buy_EV"].values for n, w in tests) / sum(w for _, w in tests))
    O.append(oof); P.append(tst); names.append(g); print(f"  {g:11s} {len(oofs)} seed(s)  OOF {roc_auc_score(y, oof):.6f}")
O, P = np.column_stack(O), np.column_stack(P)
if os.environ.get("PROBIT") == "1":   # blend in probit space: ndtri of each group's rank column, OOF and test alike (issue #1)
    from scipy.special import ndtri; O, P = ndtri(np.clip(O, 1e-6, 1 - 1e-6)), ndtri(np.clip(P, 1e-6, 1 - 1e-6))
def fit_weights(O, y):
    """Coordinate search over the simplex (each weight tried on a grid, others held, renormalised), 4 passes, then a
    Nelder-Mead polish. Nelder-Mead alone from equal weights stalls once there are more than ~10 groups (blend_v5
    first fit: 0.946346 with 16 groups, below the 13-group blend_v4 at 0.946351)."""
    n = O.shape[1]; auc = lambda w: roc_auc_score(y, O @ (w / w.sum())); w = np.ones(n) / n; best = auc(w)
    grid = [0, .01, .02, .03, .05, .07, .1, .13, .17, .22, .28, .35, .45, .6]
    for p in range(4):
        improved = False
        for j in range(n):
            for g in grid:
                w2 = w.copy(); w2[j] = g
                if w2.sum() == 0: continue
                a = auc(w2)
                if a > best + 1e-9: best, w, improved = a, w2 / w2.sum(), True
        print(f"  pass {p}: {best:.6f}", flush=True)
        if not improved: break
    res = minimize(lambda v: -auc(np.abs(v) + 1e-12), w, method="Nelder-Mead", options=dict(maxfev=500, xatol=1e-4, fatol=1e-9))
    if -res.fun > best: w, best = np.abs(res.x) + 1e-12, -res.fun
    return w / w.sum(), best
w, auc = fit_weights(O, y)
print(f"blend OOF {auc:.6f}  weights {dict(zip(names, np.round(w, 3)))}  best group {max(roc_auc_score(y, O[:, i]) for i in range(len(names))):.6f}")
np.save(f"submissions/oof_{out}.npy", O @ w); pd.DataFrame({"id": te.id, "Will_Buy_EV": r(P @ w) if os.environ.get("PROBIT") == "1" else P @ w}).to_csv(f"submissions/{out}.csv", index=False); print(f"wrote submissions/{out}.csv")
