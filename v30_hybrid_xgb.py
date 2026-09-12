"""v30: XGBoost (hist, depth 5) on the v27 hybrid frame, v7's params. Family diversity on the stronger frame.
Usage: python v30_hybrid_xgb.py [seed]"""
import sys, time, numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from features import load
from v27_hybrid import build, fold_frames, TARGET
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42; N = 5; name = f"v30_hybrid_xgb_s{SEED}"; rk = lambda v: rankdata(v) / len(v)
t = time.time(); tr, te, o, y, feats = load(); X, Xte, K, Kte = build(tr, te, o)
ref = roc_auc_score(y, np.load("submissions/oof_v27_hybrid_k5_s42.npy")); print(f"{name}: v27 ref OOF {ref:.6f}  features {X.shape[1]}", flush=True)
cv = StratifiedKFold(N, shuffle=True, random_state=SEED); oof = np.zeros(len(X)); pte = np.zeros(len(Xte)); its = []
for f, (a, b) in enumerate(cv.split(X, y)):
    A, B, C = fold_frames(X, Xte, K, Kte, y, a, b)
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, max_depth=5, learning_rate=0.05, n_estimators=6000, subsample=0.9,
                          colsample_bytree=0.6, min_child_weight=50, reg_lambda=2.0, max_cat_to_onehot=1, eval_metric="auc",
                          early_stopping_rounds=200, n_jobs=12, random_state=SEED)
    m.fit(A, y[a], eval_set=[(B, y[b])], verbose=False)
    pb = m.predict_proba(B)[:, 1]; oof[b] = pb; pte += rk(m.predict_proba(C)[:, 1]) / N; its.append(m.best_iteration)
    print(f"  fold {f}: auc {roc_auc_score(y[b], pb):.6f}  iters {its[-1]}  {time.time()-t:.0f}s", flush=True)
auc = roc_auc_score(y, oof); print(f"{name} OOF {auc:.6f}  Δ vs v27 {auc-ref:+.6f}  iters {its}  {time.time()-t:.0f}s", flush=True)
np.save(f"submissions/oof_{name}.npy", oof); pd.DataFrame({"id": te.id, TARGET: pte}).to_csv(f"submissions/{name}.csv", index=False)
