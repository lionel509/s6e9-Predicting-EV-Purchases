"""v13: full-data, multi-seed refit of the v9 pseudo-label config — the test-side swap. Trains on every train
row plus the test rows as soft labels (source = v9's own predictions, i.e. pseudo round 2), no validation, rounds =
1.15 x v9's mean best iteration, averaged over seeds. No OOF exists for a full-data model: blend with v9's OOF
weights and substitute this test column. Usage: python v13_refit.py <soft source> <rounds> <m> <params: base|slow|naji> <seeds...>"""
import sys, time, numpy as np, pandas as pd, lightgbm as lgb
from features import *
src, rounds, m, pv, seeds = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4], [int(s) for s in sys.argv[5:]] or [42]
V = {"base": LGB_PARAMS, "slow": dict(LGB_PARAMS, learning_rate=0.02),
     "naji": dict(learning_rate=0.02, max_depth=5, num_leaves=32, min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_alpha=0.071, reg_lambda=2.0, max_bin=1024, verbose=-1)}
tr, te, o, y, feats = load(); base = Base(tr, te, o, feats); Btr, Bte = base(tr), base(te)
soft = pd.read_csv(f"submissions/{src}.csv")["Will_Buy_EV"].values; assert 0.1 < soft.mean() < 0.3, "source must be probabilities, not ranks"
tenc = TE(tr, te, V5_SPECS + ROUND_SPECS, m=m); allidx = np.arange(len(tr))
# fit rows get inner-OOF encodings (as in CV); test rows the full-train encoding — identical to TE.fold with a = everything
Xa, _, Xt = tenc.fold(allidx, allidx[:1], y, Btr, Bte)
Xall = pd.concat([Xa, Xt], ignore_index=True); yall = np.r_[y, soft]
p = {k: v for k, v in V[pv].items() if k != "n_estimators"} | dict(objective="cross_entropy")
pred = np.zeros(len(te)); t = time.time()
for s in seeds:
    mdl = lgb.train(dict(p, seed=s, bagging_seed=s, feature_fraction_seed=s), lgb.Dataset(Xall, yall), num_boost_round=rounds)
    pred += mdl.predict(Xt) / len(seeds); print(f"  seed {s} done {time.time()-t:.0f}s", flush=True)
name = f"v13_refit_{src}_r{rounds}_m{m:g}_{pv}_s{len(seeds)}"
pd.DataFrame({"id": te.id, "Will_Buy_EV": pred}).to_csv(f"submissions/{name}.csv", index=False)
print(f"{name} written; corr with source test preds {np.corrcoef(pred, soft)[0,1]:.5f}  mean {pred.mean():.4f}", flush=True)
