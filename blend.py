"""Fit blend weights on saved OOF vectors (rank-space, Nelder-Mead from equal weights, maximising OOF AUC),
then apply the same weights to the matching test predictions. Usage: python blend.py name1 name2 ... [--out blend_x]"""
import sys, numpy as np, pandas as pd
from scipy.stats import rankdata
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from features import load
out = sys.argv[sys.argv.index("--out")+1] if "--out" in sys.argv else "blend"
args = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and sys.argv[i-1] != "--out"]
tr, te, o, y, feats = load()
oofs = {n: np.load(f"submissions/oof_{n}.npy") for n in args}
preds = {n: pd.read_csv(f"submissions/{n}.csv")["Will_Buy_EV"].values for n in args}
r = lambda a: rankdata(a) / len(a)
O = np.column_stack([r(oofs[n]) for n in args]); P = np.column_stack([r(preds[n]) for n in args])
for n in args: print(f"  {n:14s} OOF {roc_auc_score(y, oofs[n]):.6f}")
C = np.corrcoef(np.column_stack([oofs[n] for n in args]).T); print("OOF correlations:\n", np.round(C, 4))
f = lambda w: -roc_auc_score(y, O @ (np.abs(w) / np.abs(w).sum()))
res = minimize(f, np.ones(len(args)) / len(args), method="Nelder-Mead", options=dict(maxfev=400, xatol=1e-4, fatol=1e-8))
w = np.abs(res.x) / np.abs(res.x).sum()
print(f"blend OOF {-res.fun:.6f}  weights {dict(zip(args, np.round(w, 3)))}  best single {max(roc_auc_score(y, oofs[n]) for n in args):.6f}")
pd.DataFrame({"id": te.id, "Will_Buy_EV": P @ w}).to_csv(f"submissions/{out}.csv", index=False); np.save(f"submissions/oof_{out}.npy", O @ w)
print(f"wrote submissions/{out}.csv")
