"""v14: paired ablations against v6b, one lever each, lifted from the public LightGBMs that beat ours by 0.0001
(Naji CV 0.94587, Mizushima). Same folds, same params unless named.
  a) nested TE + counts on all 13 columns, not just income/commute/age      (Mizushima, Naji)
  b) max_bin 1024 — income has 13k distinct values, 255 bins coarsen it    (Naji)
  c) drop Number_of_Cars_Owned — not in the generator recipe                (Naji)
  d) original-dataset target mean for every column, not just income/commute (Naji)
Usage: python v14_ablate.py a b c d"""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from features import *
which = sys.argv[1:] or list("abcd")
tr, te, o, y, feats = load(); base = Base(tr, te, o, feats); skf = folds(tr, y)
ref = roc_auc_score(y, np.load("submissions/oof_v6b_round.npy")); print(f"v6b ref OOF {ref:.6f}", flush=True)
es = lambda Xb, yb: dict(eval_set=[(Xb, yb)], callbacks=[lgb.early_stopping(150, verbose=False)])
OTHER = [c for c in feats if c not in V5_SPECS]
def orig_means(df, X):
    om = (o["Will_Buy_EV"] == "Yes").mean()
    for c in OTHER:
        g = o.groupby(c)["Will_Buy_EV"].apply(lambda s: (s == "Yes").mean())
        X[c + "_orig_rate"] = df[c].map(g).fillna(om).astype(float)
    return X
for w in which:
    Btr, Bte = base(tr), base(te); specs = V5_SPECS + ROUND_SPECS; params = dict(LGB_PARAMS); name = f"v14{w}"
    if w == "a":
        specs = specs + OTHER
        synth = pd.concat([tr, te], ignore_index=True)
        for c in OTHER:
            vc = synth[c].value_counts(); Btr[c + "_cnt"] = tr[c].map(vc).astype(float); Bte[c + "_cnt"] = te[c].map(vc).astype(float)
        name += "_te_all"
    if w == "b": params["max_bin"] = 1024; name += "_bin1024"
    if w == "c": Btr = Btr.drop(columns="Number_of_Cars_Owned"); Bte = Bte.drop(columns="Number_of_Cars_Owned"); name += "_nocars"
    if w == "d": Btr, Bte = orig_means(tr, Btr), orig_means(te, Bte); name += "_origall"
    lgbm = lambda: lgb.LGBMClassifier(**params)
    _, _, auc = run_cv(lgbm, tr, te, y, Btr, Bte, specs, skf, name, es); print(f"   Δ vs v6b {auc-ref:+.6f}", flush=True)
