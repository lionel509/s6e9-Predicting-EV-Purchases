"""v24: label smoothing toward the recipe. Targets = (1-a)*y + a*Phi(buy - c), c calibrated on the fit rows, trained
with cross_entropy; the idea is to cut the variance the Gaussian wobble injects into every leaf. Validation on the
real labels. Naji-shaped params, m=1, same folds. Usage: python v24_smooth.py <a>..."""
import sys, numpy as np, pandas as pd, lightgbm as lgb, time
from scipy.stats import norm
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score
from features import *
from v10_recipe import recipe
alphas = [float(a) for a in sys.argv[1:]] or [0.1]
NAJI = dict(learning_rate=0.02, max_depth=5, num_leaves=32, min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_alpha=0.071, reg_lambda=2.0, max_bin=1024, verbose=-1)
tr, te, o, y, feats = load(); base = Base(tr, te, o, feats); Btr, Bte = base(tr), base(te); skf = folds(tr, y); s = recipe(tr).buy_score.values
ref = roc_auc_score(y, np.load("submissions/oof_v23_naji_m1.npy")); print(f"v23 (clean naji m1) ref OOF {ref:.6f}", flush=True)
tenc = TE(tr, te, V5_SPECS + ROUND_SPECS, m=1); p = dict(NAJI, objective="cross_entropy", metric="auc")
for al in alphas:
    oof = np.zeros(len(tr)); pred = np.zeros(len(te)); its = []; t = time.time()
    for a, b in skf:
        Xa, Xb, Xt = tenc.fold(a, b, y, Btr, Bte); c = brentq(lambda c: norm.cdf(s[a] - c).mean() - y[a].mean(), 0, 12)
        ya = (1 - al) * y[a] + al * norm.cdf(s[a] - c)
        d = lgb.Dataset(Xa, ya); v = lgb.Dataset(Xb, y[b], reference=d)
        mdl = lgb.train(p, d, num_boost_round=20000, valid_sets=[v], callbacks=[lgb.early_stopping(300, verbose=False)])
        oof[b] = mdl.predict(Xb); pred += mdl.predict(Xt) / 5; its.append(mdl.best_iteration)
    auc = roc_auc_score(y, oof); name = f"v24_smooth_a{al:g}"
    print(f"{name} OOF {auc:.6f}  Δ vs v23 {auc-ref:+.6f}  iters {its}  {time.time()-t:.0f}s", flush=True)
    np.save(f"submissions/oof_{name}.npy", oof); pd.DataFrame({"id": te.id, "Will_Buy_EV": pred}).to_csv(f"submissions/{name}.csv", index=False)
