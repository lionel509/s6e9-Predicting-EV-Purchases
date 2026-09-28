"""What do the Grand Prix engines (import_gp.py) add on top of our best blend? Per engine: OOF AUC, Spearman vs ours,
best 2-way probit-rank weight and the gain it buys; then greedy forward selection with each step's gain re-checked
nested (weights fitted on 4/5 of rows, scored on the 5th). Usage: python diag_gp.py <our blend name>"""
import sys, glob, numpy as np
from scipy.stats import rankdata, norm, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from features import load
tr, te, o, y, feats = load()
pr = lambda a: norm.ppf(rankdata(a) / (len(a) + 1))
base = sys.argv[1] if len(sys.argv) > 1 else "blend_v27fr_nov19"
B = pr(np.load(f"submissions/oof_{base}.npy")); b0 = roc_auc_score(y, B)
E = {f[len("submissions/oof_gp_"):-4]: pr(np.load(f)) for f in sorted(glob.glob("submissions/oof_gp_*.npy"))}
W = np.linspace(0, 0.5, 26)
def best_w(X, v, idx=None):
    yy, XX, vv = (y, X, v) if idx is None else (y[idx], X[idx], v[idx])
    s = [roc_auc_score(yy, (1 - w) * XX + w * vv) for w in W]; i = int(np.argmax(s)); return W[i], s[i]
print(f"base {base} OOF {b0:.6f}\n{'engine':24s} {'AUC':>9s} {'rho':>8s} {'w':>5s} {'gain':>10s}")
rows = []
for k, v in E.items():
    w, s = best_w(B, v); rows.append((s - b0, k)); print(f"{k:24s} {roc_auc_score(y, v):9.6f} {spearmanr(B, v).statistic:8.5f} {w:5.2f} {s - b0:+10.6f}", flush=True)
print("\ngreedy forward (in-sample / nested):")
cur, folds = B.copy(), list(StratifiedKFold(5, shuffle=True, random_state=7).split(B, y))
cand = [k for g, k in sorted(rows, reverse=True)[:12] if g > 0]
for step in range(6):
    best = max(((best_w(cur, E[k])[1], k) for k in cand), default=None)
    if not best or best[0] <= roc_auc_score(y, cur): break
    k = best[1]; nest = np.zeros(len(y))
    for a, bb in folds: w, _ = best_w(cur, E[k], a); nest[bb] = (1 - w) * cur[bb] + w * E[k][bb]
    base_n = roc_auc_score(y, cur); w, _ = best_w(cur, E[k])
    print(f"  +{k:22s} w {w:.2f}  in {best[0]:.6f} ({best[0] - base_n:+.6f})  nested {roc_auc_score(y, nest):.6f} ({roc_auc_score(y, nest) - base_n:+.6f})", flush=True)
    cur = (1 - w) * cur + w * E[k]; cand.remove(k)
