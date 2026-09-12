"""Nested stacker: can a small model that sees the OOF rank columns PLUS where the row lives (income frequency, income,
the recipe score) beat the global linear blend? The 09-06 error map showed per-band AUC from 0.943 (common incomes)
to 0.974 (rare ones), so region-dependent weights are the one thing a linear blend cannot express.
Outer 5-fold (seed 7) on the OOF rows: fit on four, score the fifth → a nested OOF AUC that is honest about the
stacker's own fit. Test column: refit on all rows with the mean best round count. Two stackers: logistic regression
(the linear baseline with the extra columns) and a shallow LightGBM. Usage: python stack.py"""
import os, numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from features import load
tr, te, o, y, feats = load(); r = lambda a: rankdata(a) / len(a)
GROUPS = {"hybrid10": ["v27_hybrid_k10_s42", "v27_hybrid_k10_s7", "v27_hybrid_k10_s2026", "v27_hybrid_k10_s101", "v27_hybrid_k10_s202"],
          "naji_pc": ["v22_pseudo_clean_naji_m1", "v22_pseudo_clean_naji_m1_f7_t0", "v22_pseudo_clean_naji_m1_f2026_t0"],
          "hybrid_xgb": ["v30_hybrid_xgb_s42_k10"], "hybrid_cat": ["v29_hybrid_cat_s42_k10"], "catboost": ["v8_catboost"], "nn": ["v12_nn_s101", "v12_nn_s202", "v12_nn_s303"],
          "pub_megayak10": ["pub_megayak10"], "pub_naji_01blend": ["pub_naji_01blend"], "pub_naji_v3": ["pub_naji_v3"], "pub_realmlp": ["pub_realmlp"], "pub_sergey": ["pub_sergey"]}
O, P, names = [], [], []
for g, oofs in GROUPS.items():
    oofs = [n for n in oofs if os.path.exists(f"submissions/oof_{n}.npy")]
    if not oofs: continue
    O.append(np.mean([r(np.load(f"submissions/oof_{n}.npy")) for n in oofs], 0)); P.append(np.mean([r(pd.read_csv(f"submissions/{n}.csv").Will_Buy_EV.values) for n in oofs], 0)); names.append(g)
O, P = np.column_stack(O), np.column_stack(P)
def side(df):
    inc = df.Annual_Income_USD.to_numpy(float); fq = pd.concat([tr, te]).Annual_Income_USD.value_counts()
    buy = 1.2 * inc / 1e5 + 0.6 * df.Environmental_Concern_Level + 2 * (df.Subsidy_Available == "Yes") - 1 * (df.Range_Anxiety_Level == "Medium") - 3 * (df.Range_Anxiety_Level == "High")
    return np.column_stack([np.log1p(df.Annual_Income_USD.map(fq).to_numpy(float)), np.log1p(inc), buy.to_numpy(float), (inc == 30000).astype(float)])
S, St = side(tr), side(te); X, Xt = np.column_stack([O, S]), np.column_stack([P, St]); side_names = ["log_fq_inc", "log_inc", "buy", "is30k"]
print("groups:", names, "| lin-blend reference OOF (equal weights):", f"{roc_auc_score(y, O.mean(1)):.6f}", flush=True)
skf = list(StratifiedKFold(5, shuffle=True, random_state=7).split(X, y))
# 1. logistic regression on rank columns (+ side features)
for cols, tag in ((O, "LR ranks only"), (X, "LR ranks + side")):
    oof = np.zeros(len(y))
    for a, b in skf:
        m = LogisticRegression(C=1.0, max_iter=500).fit(cols[a], y[a]); oof[b] = m.decision_function(cols[b])
    print(f"{tag:20s} nested OOF {roc_auc_score(y, oof):.6f}", flush=True)
# 2. shallow LightGBM stacker, inner early stopping on a slice of the fit rows
p = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=8, min_data_in_leaf=2000, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, num_threads=6, verbose=-1, seed=42)
for cols, colnames, tag in ((O, names, "LGBM ranks only"), (X, names + side_names, "LGBM ranks + side")):
    oof = np.zeros(len(y)); its = []
    for a, b in skf:
        ia, iv = next(StratifiedKFold(5, shuffle=True, random_state=1).split(a, y[a])); ia, iv = a[ia], a[iv]
        d = lgb.Dataset(cols[ia], y[ia]); v = lgb.Dataset(cols[iv], y[iv], reference=d)
        m = lgb.train(p, d, num_boost_round=3000, valid_sets=[v], callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[b] = m.predict(cols[b]); its.append(m.best_iteration)
    auc = roc_auc_score(y, oof); print(f"{tag:20s} nested OOF {auc:.6f}  iters {its}", flush=True)
    if tag == "LGBM ranks + side":
        m = lgb.train(p, lgb.Dataset(cols, y), num_boost_round=int(np.mean(its)))
        imp = dict(zip(colnames, m.feature_importance("gain").round(0))); print("  gain importance:", sorted(imp.items(), key=lambda kv: -kv[1])[:8])
        np.save("submissions/oof_stack_lgbm.npy", oof); pd.DataFrame({"id": te.id, "Will_Buy_EV": m.predict(Xt)}).to_csv("submissions/stack_lgbm.csv", index=False); print("  wrote submissions/stack_lgbm.csv")
