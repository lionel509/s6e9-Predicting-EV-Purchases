"""v29: CatBoost on the v27 hybrid frame (build + triple TE per fold). The six string cats as CatBoost categoricals,
plus income / commute / age as categorical copies (its own ordered target statistics next to the sklearn TE).
v8's params. The blend's one genuinely different estimator, now on the stronger frame. Usage: python v29_hybrid_cat.py [seed]"""
import sys, time, numpy as np, pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from features import load
from v27_hybrid import build, fold_frames, CATS, TARGET
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42; N = 5; name = f"v29_hybrid_cat_s{SEED}"; rk = lambda v: rankdata(v) / len(v)
t = time.time(); tr, te, o, y, feats = load(); X, Xte, K, Kte = build(tr, te, o)
HI = ["Annual_Income_USD", "Daily_Commute_km", "Age"]
for df, src in ((X, tr), (Xte, te)):
    for c in CATS: df[c] = df[c].astype(str)
    for c in HI: df[c + "_cat"] = src[c].astype(str).values
cat_cols = CATS + [c + "_cat" for c in HI]
ref = roc_auc_score(y, np.load("submissions/oof_v27_hybrid_k5_s42.npy")); print(f"{name}: v27 ref OOF {ref:.6f}  features {X.shape[1]}", flush=True)
cv = StratifiedKFold(N, shuffle=True, random_state=SEED); oof = np.zeros(len(X)); pte = np.zeros(len(Xte)); its = []
for f, (a, b) in enumerate(cv.split(X, y)):
    A, B, C = fold_frames(X, Xte, K, Kte, y, a, b)
    m = CatBoostClassifier(iterations=4000, learning_rate=0.06, depth=6, l2_leaf_reg=3, eval_metric="AUC", od_type="Iter", od_wait=150,
                           cat_features=cat_cols, thread_count=12, random_seed=SEED, verbose=0, allow_writing_files=False)
    m.fit(A, y[a], eval_set=(B, y[b]), use_best_model=True)
    pb = m.predict_proba(B)[:, 1]; oof[b] = pb; pte += rk(m.predict_proba(C)[:, 1]) / N; its.append(m.get_best_iteration())
    print(f"  fold {f}: auc {roc_auc_score(y[b], pb):.6f}  iters {its[-1]}  {time.time()-t:.0f}s", flush=True)
auc = roc_auc_score(y, oof); print(f"{name} OOF {auc:.6f}  Δ vs v27 {auc-ref:+.6f}  iters {its}  {time.time()-t:.0f}s", flush=True)
np.save(f"submissions/oof_{name}.npy", oof); pd.DataFrame({"id": te.id, TARGET: pte}).to_csv(f"submissions/{name}.csv", index=False)
