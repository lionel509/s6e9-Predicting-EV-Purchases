"""v19: pseudo-labelling, round two, with everything learned since v9: rounded-income TE keys (v9 had only the three
base keys), TE smoothing m from v15, optional parameter variant from v17, soft labels from a probability submission.
Validation on real rows only. Usage: python v19_pseudo2.py <source> <m> [params: base|slow|leaves32|naji] [weight]"""
import sys, numpy as np, pandas as pd, lightgbm as lgb, time
from sklearn.metrics import roc_auc_score
from features import *
src, m = sys.argv[1], float(sys.argv[2]); pv = sys.argv[3] if len(sys.argv) > 3 else "base"; wt = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
V = {"base": LGB_PARAMS,
     "slow": dict(LGB_PARAMS, learning_rate=0.02),
     "leaves32": dict(LGB_PARAMS, num_leaves=32, min_data_in_leaf=60),
     "naji": dict(learning_rate=0.02, max_depth=5, num_leaves=32, min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_alpha=0.071, reg_lambda=2.0, max_bin=1024, verbose=-1)}
tr, te, o, y, feats = load(); base = Base(tr, te, o, feats); Btr, Bte = base(tr), base(te); skf = folds(tr, y)
soft = pd.read_csv(f"submissions/{src}.csv")["Will_Buy_EV"].values; assert 0.1 < soft.mean() < 0.3, "source must be probabilities"
ref = roc_auc_score(y, np.load("submissions/oof_v6b_round.npy")); print(f"v6b ref OOF {ref:.6f}   source {src}", flush=True)
tenc = TE(tr, te, V5_SPECS + ROUND_SPECS, m=m); p = {k: v for k, v in V[pv].items() if k != "n_estimators"} | dict(objective="cross_entropy", metric="auc")
oof = np.zeros(len(tr)); pred = np.zeros(len(te)); its = []; t = time.time()
for a, b in skf:
    Xa, Xb, Xt = tenc.fold(a, b, y, Btr, Bte)
    Xall = pd.concat([Xa, Xt], ignore_index=True); yall = np.r_[y[a], soft]; wall = np.r_[np.ones(len(a)), np.full(len(te), wt)]
    dtr = lgb.Dataset(Xall, yall, weight=wall); dva = lgb.Dataset(Xb, y[b], reference=dtr)
    mdl = lgb.train(p, dtr, num_boost_round=20000, valid_sets=[dva], callbacks=[lgb.early_stopping(300, verbose=False)])
    oof[b] = mdl.predict(Xb); pred += mdl.predict(Xt) / 5; its.append(mdl.best_iteration)
auc = roc_auc_score(y, oof); name = f"v19_pseudo2_m{m:g}_{pv}"
print(f"{name} OOF {auc:.6f}  Δ vs v6b {auc-ref:+.6f}  iters {its}  {time.time()-t:.0f}s", flush=True)
np.save(f"submissions/oof_{name}.npy", oof); pd.DataFrame({"id": te.id, "Will_Buy_EV": pred}).to_csv(f"submissions/{name}.csv", index=False)
