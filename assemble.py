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
  "hybrid":   (["v27_hybrid_k5_s42"], [("v27_hybrid_k10_s42", 2), ("v27_hybrid_k5_s42", 1)]),
  "hybrid10": (["v27_hybrid_k10_s42", "v27_hybrid_k10_s7", "v27_hybrid_k10_s2026", "v27_hybrid_k10_s101", "v27_hybrid_k10_s202", "v27_hybrid_k10_s303", "v27_hybrid_k10_s404", "v27_hybrid_k10_s505"], None),   # 10-fold OOFs (like-for-like with pub_megayak10), three fold seeds   # megayak recipe on our split, OOF 0.946153; test column leans on the 10-fold run
  "hybrid_pseudo": (["v28_hybrid_pseudo_k5_s42"], None),   # leak-free two-stage pseudo on the hybrid frame
  "hybrid_m1": (["v28_hybrid_m1_k5_s42"], None), "hybrid_m1_pseudo": (["v28_hybrid_pseudo_m1_k5_s42"], None),
  "hybrid_cat": (["v29_hybrid_cat_s42_k10"], None), "hybrid_xgb": (["v30_hybrid_xgb_s42_k10"], None), "hybrid_nn": (["v31_hybrid_nn_s42"], None),
  "naji_pc":  (["v22_pseudo_clean_naji_m1", "v22_pseudo_clean_naji_m1_f7_t0", "v22_pseudo_clean_naji_m1_f2026_t0"],
               [("v13_refit_v22_pseudo_clean_naji_m1_r2290_m1_naji_s3", 3), ("v22_pseudo_clean_naji_m1", 1), ("v22_pseudo_clean_naji_m1_f7_t0", 1), ("v22_pseudo_clean_naji_m1_f2026_t0", 1)]),   # test: half full-data refit, half the 15-model fold bag
  "naji_clean": (["v23_naji_m1", "v23_naji_m1_f7_t0", "v23_naji_m1_f2026_t0"], None),
  "catboost": (["v8_catboost"], [("v26_cat_refit_r1330_s3", 1), ("v8_catboost", 1)]),   # same hedge
  "nn":       (["v12_nn_s101", "v12_nn_s202", "v12_nn_s303"], None),
  "pub_megayak10": (["pub_megayak10"], None), "pub_naji_v3": (["pub_naji_v3"], None), "pub_sergey": (["pub_sergey"], None), "pub_realmlp": (["pub_realmlp"], None), "pub_naji_01blend": (["pub_naji_01blend"], None),   # public OOF sources (import_public.py)
  "pub_mega_A": (["pub_mega_A"], None), "pub_mega_B": (["pub_mega_B"], None), "pub_mega_C": (["pub_mega_C"], None), "pub_mega_D": (["pub_mega_D"], None),   # megayak four views, 10-fold s42
  "pub_realmlp2": (["pub_realmlp2"], None), "pub_naji_xgb": (["pub_naji_xgb"], None),   # yekenot RealMLP (2026-09-13 version), Naji XGB 10-fold
  "viewC": (["v33_viewC_k10_s42"], None), "viewD": (["v33_viewD_k10_s42", "v33_viewD_k10_s7", "v33_viewD_k10_s2026", "v33_viewD_k10_s101", "v33_viewD_k10_s202"], None), "viewE": (["v33_viewE_k10_s42"], None), "viewF": (["v33_viewF_k10_s42"], None), "viewD_xgb": (["v33_viewD_xgb_k10_s42"], None), "viewD_init50": (["v33_viewD_initinc50_k10_s42"], None), "viewD_init500": (["v33_viewD_initinc500_k10_s42"], None), "viewC_xgb": (["v33_viewC_xgb_k10_s42"], None),   # megayak views C / D and our //25 ladder E, on our pipeline (v33)
  "init": (["v34_init_inc_exact_k10_s42", "v34_init_inc_exact_k10_s7", "v34_init_inc_exact_k10_s2026", "v34_init_inc_exact_k10_s101", "v34_init_inc_exact_k10_s202"], None), "realmlp_own": (["v35_realmlp_k5_s42"], None), "realmlp_e3": (["v35_realmlp_k5_s42_e3"], None), "realmlp_own10": (["v35_realmlp_k10_s42", "v35_realmlp_k10_s7", "v35_realmlp_k10_s2026", "v35_realmlp_k10_s101", "v35_realmlp_k10_s202"], None), "init_drop": (["v34_init_inc_exact_drop_k10_s42"], None), "init100": (["v34_init_inc100_k10_s42", "v34_init_inc100_k10_s7", "v34_init_inc100_k10_s2026", "v34_init_inc100_k10_s101", "v34_init_inc100_k10_s202"], None), "init1000": (["v34_init_inc1000_k10_s42"], None),   # init_score views (v34)
  "rank": (["v32_rank_p8_k5_s42"], None), "linear": (["v32_linear_l0_k5_s42"], None), "extra": (["v32_extra_k10_s42"], None),   # v32 variants that survive
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
np.save(f"submissions/oof_{out}.npy", O @ w); pd.DataFrame({"id": te.id, "Will_Buy_EV": P @ w}).to_csv(f"submissions/{out}.csv", index=False); print(f"wrote submissions/{out}.csv")
