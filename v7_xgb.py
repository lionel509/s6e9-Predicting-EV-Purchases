"""v7: XGBoost on the v5 feature set (Base + nested TE of income/commute/age). Diversity for the blend."""
import numpy as np, xgboost as xgb
from sklearn.metrics import roc_auc_score
from features import *
tr, te, o, y, feats = load(); base = Base(tr, te, o, feats); Btr, Bte = base(tr), base(te); skf = folds(tr, y)
ref = roc_auc_score(y, np.load("submissions/oof_v5_te.npy"))
model = lambda: xgb.XGBClassifier(tree_method="hist", enable_categorical=True, max_depth=5, learning_rate=0.05, n_estimators=4000,
                                  subsample=0.9, colsample_bytree=0.6, min_child_weight=50, reg_lambda=2.0, max_cat_to_onehot=1,
                                  eval_metric="auc", early_stopping_rounds=150, n_jobs=12, random_state=42)
es = lambda Xb, yb: dict(eval_set=[(Xb, yb)], verbose=False)
_, _, auc = run_cv(model, tr, te, y, Btr, Bte, V5_SPECS, skf, "v7_xgb", es); print(f"   Δ vs v5 {auc-ref:+.6f}", flush=True)
