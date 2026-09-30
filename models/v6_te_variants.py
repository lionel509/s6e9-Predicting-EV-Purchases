"""v6: target-encoding variants on top of v5 — pair keys, rounded-income keys, both. Paired vs v5."""
import numpy as np, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from features import *
tr, te, o, y, feats = load(); base = Base(tr, te, o, feats); Btr, Bte = base(tr), base(te); skf = folds(tr, y)
ref = roc_auc_score(y, np.load("submissions/oof_v5_te.npy")); print(f"v5 ref OOF {ref:.6f}")
lgbm = lambda: lgb.LGBMClassifier(**LGB_PARAMS)
es = lambda Xb, yb: dict(eval_set=[(Xb, yb)], callbacks=[lgb.early_stopping(150, verbose=False)])
for name, specs in {"v6a_pairs": V5_SPECS + PAIR_SPECS, "v6b_round": V5_SPECS + ROUND_SPECS, "v6c_both": V5_SPECS + PAIR_SPECS + ROUND_SPECS}.items():
    _, _, auc = run_cv(lgbm, tr, te, y, Btr, Bte, specs, skf, name, es); print(f"   Δ vs v5 {auc-ref:+.6f}", flush=True)
