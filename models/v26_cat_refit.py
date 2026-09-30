"""v26: full-data CatBoost refit of the v8 config (its own ordered target statistics; the blend's one different
estimator, 18% of run 38). 1330 rounds = 1.15 x v8's mean best iteration; seeds averaged; no OOF — test-side swap
for v8_catboost. Usage: python v26_cat_refit.py <rounds> <seeds...>"""
import sys, time, numpy as np, pandas as pd
from catboost import CatBoostClassifier
from features import *
rounds, seeds = int(sys.argv[1]), [int(s) for s in sys.argv[2:]] or [42]
tr, te, o, y, feats = load(); base = Base(tr, te, o, feats)
def frame(df):
    X = base(df)
    for c in base.cats: X[c] = X[c].astype(str)
    for c in ["Annual_Income_USD", "Daily_Commute_km", "Age"]: X[c + "_cat"] = df[c].astype(str)
    return X
Btr, Bte = frame(tr), frame(te); cat_cols = base.cats + ["Annual_Income_USD_cat", "Daily_Commute_km_cat", "Age_cat"]
pred = np.zeros(len(te)); t = time.time()
for s in seeds:
    m = CatBoostClassifier(iterations=rounds, learning_rate=0.06, depth=6, l2_leaf_reg=3, cat_features=cat_cols, thread_count=12, random_seed=s, verbose=0, allow_writing_files=False)
    m.fit(Btr, y); pred += m.predict_proba(Bte)[:, 1] / len(seeds); print(f"  seed {s} done {time.time()-t:.0f}s", flush=True)
name = f"v26_cat_refit_r{rounds}_s{len(seeds)}"; pd.DataFrame({"id": te.id, "Will_Buy_EV": pred}).to_csv(f"submissions/{name}.csv", index=False)
src = pd.read_csv("submissions/v8_catboost.csv")["Will_Buy_EV"].values; print(f"{name} written; corr with v8 fold-average {np.corrcoef(pred, src)[0,1]:.5f}", flush=True)
