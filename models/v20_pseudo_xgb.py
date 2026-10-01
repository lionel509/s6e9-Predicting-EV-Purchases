"""v20: XGBoost on the v19 frame (Base + nested TE m=1 on base + rounded keys) with the test rows as soft labels.
Family diversity at v19's strength. Usage: python v20_pseudo_xgb.py <source> <m>"""
import sys, numpy as np, pandas as pd, xgboost as xgb, time
from sklearn.metrics import roc_auc_score
from features import *
src, m = sys.argv[1], float(sys.argv[2])
tr, te, o, y, feats = load(); base = Base(tr, te, o, feats); Btr, Bte = base(tr), base(te); skf = folds(tr, y)
soft = pd.read_csv(f"submissions/{src}.csv")["Will_Buy_EV"].values; assert 0.1 < soft.mean() < 0.3
ref = roc_auc_score(y, np.load("submissions/oof_v6b_round.npy")); print(f"v6b ref OOF {ref:.6f}   source {src}", flush=True)
tenc = TE(tr, te, V5_SPECS + ROUND_SPECS, m=m)
P = dict(tree_method="hist", max_depth=5, eta=0.05, subsample=0.9, colsample_bytree=0.6, min_child_weight=50, reg_lambda=2.0, max_cat_to_onehot=1,
         objective="binary:logistic", eval_metric="auc", nthread=12, seed=42)
oof = np.zeros(len(tr)); pred = np.zeros(len(te)); its = []; t = time.time()
for a, b in skf:
    Xa, Xb, Xt = tenc.fold(a, b, y, Btr, Bte)
    Xall = pd.concat([Xa, Xt], ignore_index=True); yall = np.r_[y[a], soft]
    dtr = xgb.DMatrix(Xall, yall, enable_categorical=True); dva = xgb.DMatrix(Xb, y[b], enable_categorical=True); dte = xgb.DMatrix(Xt, enable_categorical=True)
    mdl = xgb.train(P, dtr, num_boost_round=20000, evals=[(dva, "va")], early_stopping_rounds=300, verbose_eval=False)
    oof[b] = mdl.predict(dva, iteration_range=(0, mdl.best_iteration + 1)); pred += mdl.predict(dte, iteration_range=(0, mdl.best_iteration + 1)) / 5; its.append(mdl.best_iteration)
auc = roc_auc_score(y, oof); name = f"v20_pseudo_xgb_m{m:g}"
print(f"{name} OOF {auc:.6f}  Δ vs v6b {auc-ref:+.6f}  iters {its}  {time.time()-t:.0f}s", flush=True)
np.save(f"submissions/oof_{name}.npy", oof); pd.DataFrame({"id": te.id, "Will_Buy_EV": pred}).to_csv(f"submissions/{name}.csv", index=False)
