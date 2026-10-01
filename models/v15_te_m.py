"""v15: target-encoding smoothing, paired vs v6b (m = 20). Naji runs two encoders side by side (sklearn 'auto'
and 10); Mizushima uses alpha 20 on every column. Variants: m = 5, m = 50, and dual (m = 5 and m = 50 as separate
columns). Same folds and params. Usage: python v15_te_m.py 5 50 dual"""
import sys, time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from features import *
which = sys.argv[1:] or ["5", "50", "dual"]
tr, te, o, y, feats = load(); base = Base(tr, te, o, feats); Btr, Bte = base(tr), base(te); skf = folds(tr, y)
ref = roc_auc_score(y, np.load("submissions/oof_v6b_round.npy")); print(f"v6b ref OOF {ref:.6f}", flush=True)
specs = V5_SPECS + ROUND_SPECS
for w in which:
    ms = [5, 50] if w == "dual" else [int(w)]
    encs = [TE(tr, te, specs, m=m) for m in ms]
    oof = np.zeros(len(tr)); pred = np.zeros(len(te)); its = []; t = time.time()
    for a, b in skf:
        Xa, Xb, Xt = encs[0].fold(a, b, y, Btr, Bte)
        for e, m in zip(encs[1:], ms[1:]):
            Ya, Yb, Yt = e.fold(a, b, y, Btr, Bte); tecols = [c for c in Ya.columns if c.endswith("_te")]
            for c in tecols: Xa[f"{c}{m}"], Xb[f"{c}{m}"], Xt[f"{c}{m}"] = Ya[c].values, Yb[c].values, Yt[c].values
        mdl = lgb.LGBMClassifier(**LGB_PARAMS); mdl.fit(Xa, y[a], eval_set=[(Xb, y[b])], callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[b] = mdl.predict_proba(Xb)[:, 1]; pred += mdl.predict_proba(Xt)[:, 1] / 5; its.append(mdl.best_iteration_)
    auc = roc_auc_score(y, oof); name = f"v15_m{w}"
    print(f"{name:12s} OOF {auc:.6f}  iters {its}  {time.time()-t:.0f}s\n   Δ vs v6b {auc-ref:+.6f}", flush=True)
    np.save(f"submissions/oof_{name}.npy", oof); pd.DataFrame({"id": te.id, "Will_Buy_EV": pred}).to_csv(f"submissions/{name}.csv", index=False)
