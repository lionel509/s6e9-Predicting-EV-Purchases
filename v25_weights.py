"""v25: "weigh the categories" — three literal readings, paired vs run 36 (clean Naji-shape, m=1, no pseudo):
  pos2 / pos05  : scale_pos_weight 2 / 0.5 (class weighting)
  cat10 / cat50 : income, commute, age ALSO passed as native LightGBM categoricals with cat_smooth 10 / 50,
                  cat_l2 10, max_cat_threshold 64 — LightGBM's own smoothed per-category statistics next to the TE
  wfreq         : row weight = 1 / sqrt(income frequency) — rare income values count more per row
Usage: python v25_weights.py pos2 pos05 cat10 cat50 wfreq"""
import sys, time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from features import *
which = sys.argv[1:] or ["pos2", "pos05", "cat10", "cat50", "wfreq"]
NAJI = dict(learning_rate=0.02, max_depth=5, num_leaves=32, min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_alpha=0.071, reg_lambda=2.0, max_bin=1024, verbose=-1)
tr, te, o, y, feats = load(); base = Base(tr, te, o, feats); Btr, Bte = base(tr), base(te); skf = folds(tr, y)
ref = roc_auc_score(y, np.load("submissions/oof_v23_naji_m1.npy")); print(f"v23 ref OOF {ref:.6f}", flush=True)
tenc = TE(tr, te, V5_SPECS + ROUND_SPECS, m=1); HI = ["Annual_Income_USD", "Daily_Commute_km", "Age"]
freq = pd.concat([tr, te]).Annual_Income_USD.value_counts(); wrow = 1 / np.sqrt(tr.Annual_Income_USD.map(freq).to_numpy())
for w in which:
    p = dict(NAJI, objective="binary", metric="auc"); oof = np.zeros(len(tr)); pred = np.zeros(len(te)); its = []; t = time.time()
    if w == "pos2": p["scale_pos_weight"] = 2.0
    if w == "pos05": p["scale_pos_weight"] = 0.5
    if w.startswith("cat"): p.update(cat_smooth=float(w[3:]), cat_l2=10.0, max_cat_threshold=64, min_data_per_group=50)
    for a, b in skf:
        Xa, Xb, Xt = tenc.fold(a, b, y, Btr, Bte)
        if w.startswith("cat"):
            for c in HI:
                cats = pd.Categorical(pd.concat([tr[c], te[c]])).categories
                for X, df in ((Xa, tr.iloc[a]), (Xb, tr.iloc[b]), (Xt, te)): X[c + "_cat"] = pd.Categorical(df[c].to_numpy(), categories=cats)
        d = lgb.Dataset(Xa, y[a], weight=wrow[a] if w == "wfreq" else None); v = lgb.Dataset(Xb, y[b], reference=d)
        mdl = lgb.train(p, d, num_boost_round=20000, valid_sets=[v], callbacks=[lgb.early_stopping(300, verbose=False)])
        oof[b] = mdl.predict(Xb); pred += mdl.predict(Xt) / 5; its.append(mdl.best_iteration)
    auc = roc_auc_score(y, oof); name = f"v25_{w}"
    print(f"{name:12s} OOF {auc:.6f}  Δ vs v23 {auc-ref:+.6f}  iters {its}  {time.time()-t:.0f}s", flush=True)
    np.save(f"submissions/oof_{name}.npy", oof); pd.DataFrame({"id": te.id, "Will_Buy_EV": pred}).to_csv(f"submissions/{name}.csv", index=False)
