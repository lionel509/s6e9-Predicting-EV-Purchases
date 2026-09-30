"""v9: pseudo-labelling. Test rows join the training folds with soft labels taken from a given submission
(LightGBM `cross_entropy` accepts probabilities). Validation stays on real train rows only.
Usage: python v9_pseudo.py <source submission name> [weight]"""
import sys, numpy as np, pandas as pd, lightgbm as lgb, time
from sklearn.metrics import roc_auc_score
from features import *
src = sys.argv[1]; wt = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
tr, te, o, y, feats = load(); base = Base(tr, te, o, feats); Btr, Bte = base(tr), base(te); skf = folds(tr, y)
soft = pd.read_csv(f"submissions/{src}.csv")["Will_Buy_EV"].values
ref = roc_auc_score(y, np.load(f"submissions/oof_{src}.npy")); print(f"source {src} OOF {ref:.6f}")
tenc = TE(tr, te, V5_SPECS); p = {k: v for k, v in LGB_PARAMS.items() if k != "n_estimators"}; p["objective"] = "cross_entropy"
oof = np.zeros(len(tr)); pred = np.zeros(len(te)); its = []; t = time.time()
for a, b in skf:
    Xa, Xb, Xt = tenc.fold(a, b, y, Btr, Bte)
    Xall = pd.concat([Xa, Xt], ignore_index=True); yall = np.r_[y[a], soft]; wall = np.r_[np.ones(len(a)), np.full(len(te), wt)]
    dtr = lgb.Dataset(Xall, yall, weight=wall); dva = lgb.Dataset(Xb, y[b], reference=dtr)
    m = lgb.train(dict(p, metric="auc"), dtr, num_boost_round=4000, valid_sets=[dva], callbacks=[lgb.early_stopping(150, verbose=False)])
    oof[b] = m.predict(Xb); pred += m.predict(Xt) / 5; its.append(m.best_iteration)
auc = roc_auc_score(y, oof); name = f"v9_pseudo_{src}"
print(f"{name} OOF {auc:.6f}  Δ vs source {auc-ref:+.6f}  iters {its}  {time.time()-t:.0f}s")
np.save(f"submissions/oof_{name}.npy", oof); pd.DataFrame({"id": te.id, "Will_Buy_EV": pred}).to_csv(f"submissions/{name}.csv", index=False)
