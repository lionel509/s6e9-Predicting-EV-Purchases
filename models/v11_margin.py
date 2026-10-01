"""v11: the generator recipe as a base margin (LightGBM init_score), Deotte's Model 2. Every row starts from
logit(Phi(buy_score - 5.5)) and the trees learn only the corrections. Two variants on the same folds as v6b:
 a) v6b features + margin        b) v6b features + recipe columns + margin. Paired vs v6b."""
import numpy as np, pandas as pd, lightgbm as lgb, time
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from features import *
from v10_recipe import recipe

def margin(df):
    p = np.clip(norm.cdf(recipe(df).buy_score.values - 5.5), 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
sig = lambda z: 1 / (1 + np.exp(-z))

tr, te, o, y, feats = load(); base = Base(tr, te, o, feats); skf = folds(tr, y)
mtr, mte = margin(tr), margin(te)
ref = roc_auc_score(y, np.load("submissions/oof_v6b_round.npy")); print(f"v6b ref OOF {ref:.6f}")
p = {k: v for k, v in LGB_PARAMS.items() if k != "n_estimators"} | dict(objective="binary", metric="auc")
for name, Btr, Bte in [("v11a_margin", base(tr), base(te)), ("v11b_margin_recipe", base(tr).join(recipe(tr)), base(te).join(recipe(te)))]:
    tenc = TE(tr, te, V5_SPECS + ROUND_SPECS); oof = np.zeros(len(tr)); pred = np.zeros(len(te)); its = []; t = time.time()
    for a, b in skf:
        Xa, Xb, Xt = tenc.fold(a, b, y, Btr, Bte)
        dtr = lgb.Dataset(Xa, y[a], init_score=mtr[a]); dva = lgb.Dataset(Xb, y[b], init_score=mtr[b], reference=dtr)
        m = lgb.train(p, dtr, num_boost_round=4000, valid_sets=[dva], callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[b] = sig(m.predict(Xb, raw_score=True) + mtr[b]); pred += sig(m.predict(Xt, raw_score=True) + mte) / 5; its.append(m.best_iteration)
    auc = roc_auc_score(y, oof)
    print(f"{name:12s} OOF {auc:.6f}  iters {its}  {time.time()-t:.0f}s\n   Δ vs v6b {auc-ref:+.6f}", flush=True)
    np.save(f"submissions/oof_{name}.npy", oof); pd.DataFrame({"id": te.id, "Will_Buy_EV": pred}).to_csv(f"submissions/{name}.csv", index=False)
