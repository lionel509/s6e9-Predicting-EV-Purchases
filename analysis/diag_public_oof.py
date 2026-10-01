"""Diagnostic: what do the public OOF files add to our blend? Loads every public OOF/test pair in data/public/ plus our
best leak-free OOFs, prints AUC and Spearman vs our blend, then fits rank-blend weights (Nelder-Mead) on subsets —
in-sample AND nested (weights fitted on 4/5 of rows, scored on the 5th) so the gain is not the weight-fit's own noise.
Usage: python diag_public_oof.py"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "models"))   # models/ holds features + the v* scripts
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from features import load
tr, te, o, y, feats = load(); r = lambda a: rankdata(a) / len(a)
P = "data/public/"
SRC = {  # name: (oof array, test array)
  "ours_blend_v2": (np.load("submissions/oof_blend_v2.npy"), pd.read_csv("submissions/blend_v2.csv").Will_Buy_EV.values),
  "ours_v22pc":    (np.load("submissions/oof_v22_pseudo_clean_naji_m1.npy"), pd.read_csv("submissions/v22_pseudo_clean_naji_m1.csv").Will_Buy_EV.values),
  "ours_cat":      (np.load("submissions/oof_v8_catboost.npy"), pd.read_csv("submissions/v8_catboost.csv").Will_Buy_EV.values),
  "ours_nn":       (np.load("submissions/oof_v12_nn_s101.npy"), pd.read_csv("submissions/v12_nn_s101.csv").Will_Buy_EV.values),
  "megayak10":     (np.load(P + "megayak/oof_hybrid.npy"), np.load(P + "megayak/test_hybrid.npy")),
  "naji_v3":       (pd.read_csv(P + "naji_v3/Pure LGBM_V3_oof.csv").OOF_Pred.values, pd.read_csv(P + "naji_v3/Pure LGBM_V3_test.csv").Will_Buy_EV.values),
  "sergey_focal":  (pd.read_csv(P + "naji_v3/Sergey_LGBM_oof.csv").Will_Buy_EV.values, pd.read_csv(P + "naji_v3/Sergey_LGBM_submission.csv").Will_Buy_EV.values),
  "naji_01blend":  (pd.read_csv(P + "naji_v3/01_blend_oof.csv").Will_Buy_EV.values, pd.read_csv(P + "naji_v3/01_submission.csv").Will_Buy_EV.values),
  "realmlp":       (pd.read_csv(P + "realmlp/oof_preds.csv").Will_Buy_EV.values, pd.read_csv(P + "realmlp/submission.csv").Will_Buy_EV.values),
}
for k, (a, b) in SRC.items(): assert len(a) == len(y) and len(b) == len(te), k
R = {k: r(a) for k, (a, _) in SRC.items()}
print(f"{'source':14s} {'OOF AUC':>9s} {'ρ vs blend_v2':>14s} {'ρ vs megayak':>13s}")
for k in SRC: print(f"{k:14s} {roc_auc_score(y, R[k]):9.6f} {spearmanr(R[k], R['ours_blend_v2']).statistic:14.5f} {spearmanr(R[k], R['megayak10']).statistic:13.5f}")
def fit(names, idx=None):
    O = np.column_stack([R[n] for n in names]); yy = y if idx is None else y[idx]; OO = O if idx is None else O[idx]
    f = lambda w: -roc_auc_score(yy, OO @ (np.abs(w) / np.abs(w).sum()))
    res = minimize(f, np.ones(len(names)) / len(names), method="Nelder-Mead", options=dict(maxfev=400, xatol=1e-4, fatol=1e-9))
    return np.abs(res.x) / np.abs(res.x).sum(), -res.fun
def nested(names):
    O = np.column_stack([R[n] for n in names]); pred = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=7).split(O, y):
        w, _ = fit(names, a); pred[b] = O[b] @ w
    return roc_auc_score(y, pred)
SETS = [["ours_blend_v2", "megayak10"], ["ours_blend_v2", "naji_v3"], ["ours_blend_v2", "megayak10", "naji_v3"],
        ["ours_blend_v2", "megayak10", "naji_v3", "sergey_focal"], ["ours_blend_v2", "megayak10", "naji_v3", "sergey_focal", "realmlp"],
        ["ours_v22pc", "ours_cat", "ours_nn", "megayak10", "naji_v3", "sergey_focal", "realmlp"],
        ["megayak10", "naji_v3", "sergey_focal", "realmlp"], ["megayak10", "naji_01blend", "realmlp"]]
print("\nblend fits (in-sample OOF / nested OOF):")
for s in SETS:
    w, auc = fit(s); print(f"  {auc:.6f} / {nested(s):.6f}  " + "  ".join(f"{n}={ww:.2f}" for n, ww in zip(s, w)), flush=True)
