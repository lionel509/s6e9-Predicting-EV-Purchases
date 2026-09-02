"""v8: CatBoost with income, commute and age declared categorical (ordered target statistics — its own
implementation of the v5 lever) alongside their numeric copies, Base counts and original lookup. No hand TE."""
import numpy as np, pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from features import *
tr, te, o, y, feats = load(); base = Base(tr, te, o, feats); skf = folds(tr, y)
ref = roc_auc_score(y, np.load("submissions/oof_v5_te.npy"))
def frame(df):
    X = base(df)
    for c in base.cats: X[c] = X[c].astype(str)
    for c in ["Annual_Income_USD", "Daily_Commute_km", "Age"]: X[c + "_cat"] = df[c].astype(str)
    return X
Btr, Bte = frame(tr), frame(te)
cat_cols = base.cats + ["Annual_Income_USD_cat", "Daily_Commute_km_cat", "Age_cat"]
model = lambda: CatBoostClassifier(iterations=4000, learning_rate=0.06, depth=6, l2_leaf_reg=3, eval_metric="AUC", od_type="Iter", od_wait=150,
                                   cat_features=cat_cols, thread_count=12, random_seed=42, verbose=0, allow_writing_files=False)
es = lambda Xb, yb: dict(eval_set=(Xb, yb), use_best_model=True)
_, _, auc = run_cv(model, tr, te, y, Btr, Bte, None, skf, "v8_catboost", es); print(f"   Δ vs v5 {auc-ref:+.6f}", flush=True)
