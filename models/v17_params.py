"""v17: parameter variants on the v6b frame. The 18-leaf / 152-min-leaf setting was tuned before target encoding
existed (trees now stop at ~600 rounds). Naji's LightGBM (CV 0.94587) runs lr 0.02, 32 leaves, depth 5, min_child 10,
colsample 0.3, es 500; Mizushima 63 leaves / min_child 80 / 1200 rounds. Paired vs v6b, same folds.
Usage: python v17_params.py slow leaves32 naji"""
import sys, numpy as np, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from features import *
which = sys.argv[1:] or ["slow", "leaves32", "naji"]
tr, te, o, y, feats = load(); base = Base(tr, te, o, feats); Btr, Bte = base(tr), base(te); skf = folds(tr, y)
ref = roc_auc_score(y, np.load("submissions/oof_v6b_round.npy")); print(f"v6b ref OOF {ref:.6f}", flush=True)
V = {"slow":     dict(LGB_PARAMS, learning_rate=0.02, n_estimators=20000),
     "leaves32": dict(LGB_PARAMS, num_leaves=32, min_data_in_leaf=60, n_estimators=20000),
     "naji":     dict(learning_rate=0.02, max_depth=5, num_leaves=32, min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.3,
                      reg_alpha=0.071, reg_lambda=2.0, max_bin=1024, n_estimators=20000, verbose=-1)}
for w in which:
    es = lambda Xb, yb: dict(eval_set=[(Xb, yb)], callbacks=[lgb.early_stopping(300 if w != "naji" else 500, verbose=False)])
    _, _, auc = run_cv(lambda: lgb.LGBMClassifier(**V[w]), tr, te, y, Btr, Bte, V5_SPECS + ROUND_SPECS, skf, f"v17_{w}", es); print(f"   Δ vs v6b {auc-ref:+.6f}", flush=True)
