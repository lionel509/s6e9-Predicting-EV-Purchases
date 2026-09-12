"""Final assembly. Each component is a GROUP: a list of OOF names rank-averaged into one signal, with an optional
test-column override (a refit CSV, or several averaged with weights). Weights are fitted on the leak-free OOFs by
Nelder-Mead from equal weights, then applied to the matching test signals. Writes submissions/<out>.csv + OOF.
Usage: python assemble.py <out>   (edit GROUPS below)"""
import sys, numpy as np, pandas as pd
from scipy.stats import rankdata
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from features import load
out = sys.argv[1] if len(sys.argv) > 1 else "blend_v2"
GROUPS = {
  # name: (oof names to rank-average, test csvs to average (name, weight) — None = the same names' own test columns)
  "hybrid":   (["v27_hybrid_k5_s42"], None),   # megayak recipe on our split, OOF 0.946153
  "naji_pc":  (["v22_pseudo_clean_naji_m1", "v22_pseudo_clean_naji_m1_f7_t0", "v22_pseudo_clean_naji_m1_f2026_t0"],
               [("v13_refit_v22_pseudo_clean_naji_m1_r2290_m1_naji_s3", 3), ("v22_pseudo_clean_naji_m1", 1), ("v22_pseudo_clean_naji_m1_f7_t0", 1), ("v22_pseudo_clean_naji_m1_f2026_t0", 1)]),   # test: half full-data refit, half the 15-model fold bag
  "naji_clean": (["v23_naji_m1", "v23_naji_m1_f7_t0", "v23_naji_m1_f2026_t0"], None),
  "catboost": (["v8_catboost"], [("v26_cat_refit_r1330_s3", 1), ("v8_catboost", 1)]),   # same hedge
  "nn":       (["v12_nn_s101", "v12_nn_s202", "v12_nn_s303"], None),
  "pub_megayak10": (["pub_megayak10"], None), "pub_naji_v3": (["pub_naji_v3"], None), "pub_sergey": (["pub_sergey"], None), "pub_realmlp": (["pub_realmlp"], None),   # public OOF sources (import_public.py)
  "m1":       (["v15_m1"], None), "slow": (["v17_slow"], None), "v6b": (["v6b_round"], None), "xgb": (["v7_xgb"], None),
}
tr, te, o, y, feats = load(); r = lambda a: rankdata(a) / len(a)
import os
O, P, names = [], [], []
for g, (oofs, tests) in GROUPS.items():
    oofs = [n for n in oofs if os.path.exists(f"submissions/oof_{n}.npy")]
    if not oofs: print(f"  {g}: no OOF files yet, skipped"); continue
    oof = np.mean([r(np.load(f"submissions/oof_{n}.npy")) for n in oofs], 0)
    if tests is None: tst = np.mean([r(pd.read_csv(f"submissions/{n}.csv")["Will_Buy_EV"].values) for n in oofs], 0)
    else:
        tests = [(n, w) for n, w in tests if os.path.exists(f"submissions/{n}.csv")]
        if not tests: tst = np.mean([r(pd.read_csv(f"submissions/{n}.csv")["Will_Buy_EV"].values) for n in oofs], 0); print(f"  {g}: refit missing, using fold-average test columns")
        else: tst = r(sum(w * pd.read_csv(f"submissions/{n}.csv")["Will_Buy_EV"].values for n, w in tests) / sum(w for _, w in tests))
    O.append(oof); P.append(tst); names.append(g); print(f"  {g:11s} {len(oofs)} seed(s)  OOF {roc_auc_score(y, oof):.6f}")
O, P = np.column_stack(O), np.column_stack(P)
f = lambda w: -roc_auc_score(y, O @ (np.abs(w) / np.abs(w).sum()))
res = minimize(f, np.ones(len(names)) / len(names), method="Nelder-Mead", options=dict(maxfev=600, xatol=1e-4, fatol=1e-9))
w = np.abs(res.x) / np.abs(res.x).sum(); auc = -res.fun
print(f"blend OOF {auc:.6f}  weights {dict(zip(names, np.round(w, 3)))}  best group {max(roc_auc_score(y, O[:, i]) for i in range(len(names))):.6f}")
np.save(f"submissions/oof_{out}.npy", O @ w); pd.DataFrame({"id": te.id, "Will_Buy_EV": P @ w}).to_csv(f"submissions/{out}.csv", index=False); print(f"wrote submissions/{out}.csv")
