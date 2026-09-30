"""v16: residual target encoding. The generator recipe (Deotte) explains AUC 0.938 with no parameters, so a plain
per-value target mean for income is mostly the recipe restated plus noise (~50 rows per value, SE ~0.06). Encode
the residual instead: enc(v) = sum(y_i - p_i) / (n_v + m) over rows with value v, p_i = recipe probability
calibrated on the fit rows (one threshold, matched to the fit-row base rate). Same nesting as TE. Columns are
ADDED to v6b's frame (plain TE kept). Paired vs v6b."""
import numpy as np, pandas as pd, lightgbm as lgb, time
from scipy.stats import norm
from scipy.optimize import brentq
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from features import *
from features import _fit
from v10_recipe import recipe

class ResidTE(TE):
    def fold(self, a, b, y, s, Btr, Bte):
        """s = buy_score per train/test row (dict with 'tr','te'). Calibrate c on fit rows so mean Phi(s-c) = mean y."""
        ya = y[a]; sa = s["tr"][a]
        c = brentq(lambda c: norm.cdf(sa - c).mean() - ya.mean(), 0, 12)
        p_tr = norm.cdf(s["tr"] - c); r = y - p_tr
        Xa, Xb, Xt = Btr.iloc[a].copy(), Btr.iloc[b].copy(), Bte.copy()
        inner = list(StratifiedKFold(self.inner, shuffle=True, random_state=self.seed).split(a, ya))
        for nm in self.ktr:
            ka = self.ktr[nm].iloc[a]; ra = r[a]; col = np.zeros(len(a))
            for ia, ib in inner:
                col[ib] = ka.iloc[ib].map(_fit(ka.iloc[ia], ra[ia], 0.0, self.m)).fillna(0.0).values
            Xa[nm + "_res"] = col; full = _fit(ka, ra, 0.0, self.m)
            Xb[nm + "_res"] = self.ktr[nm].iloc[b].map(full).fillna(0.0).values
            Xt[nm + "_res"] = self.kte[nm].map(full).fillna(0.0).values
        return Xa, Xb, Xt, c

tr, te, o, y, feats = load(); base = Base(tr, te, o, feats); Btr, Bte = base(tr), base(te); skf = folds(tr, y)
s = {"tr": recipe(tr).buy_score.values, "te": recipe(te).buy_score.values}
ref = roc_auc_score(y, np.load("submissions/oof_v6b_round.npy")); print(f"v6b ref OOF {ref:.6f}", flush=True)
specs = V5_SPECS + ROUND_SPECS; plain = TE(tr, te, specs); resid = ResidTE(tr, te, specs)
oof = np.zeros(len(tr)); pred = np.zeros(len(te)); its = []; t = time.time()
for a, b in skf:
    Xa, Xb, Xt = plain.fold(a, b, y, Btr, Bte); Ra, Rb, Rt, c = resid.fold(a, b, y, s, Btr, Bte)
    for col in [x for x in Ra.columns if x.endswith("_res")]: Xa[col], Xb[col], Xt[col] = Ra[col].values, Rb[col].values, Rt[col].values
    m = lgb.LGBMClassifier(**LGB_PARAMS); m.fit(Xa, y[a], eval_set=[(Xb, y[b])], callbacks=[lgb.early_stopping(150, verbose=False)])
    oof[b] = m.predict_proba(Xb)[:, 1]; pred += m.predict_proba(Xt)[:, 1] / 5; its.append(m.best_iteration_)
    print(f"  fold c={c:.3f} auc {roc_auc_score(y[b], oof[b]):.6f}", flush=True)
auc = roc_auc_score(y, oof); name = "v16_resid_te"
print(f"{name:12s} OOF {auc:.6f}  iters {its}  {time.time()-t:.0f}s\n   Δ vs v6b {auc-ref:+.6f}", flush=True)
np.save(f"submissions/oof_{name}.npy", oof); pd.DataFrame({"id": te.id, "Will_Buy_EV": pred}).to_csv(f"submissions/{name}.csv", index=False)
