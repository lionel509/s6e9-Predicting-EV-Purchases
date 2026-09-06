"""v22: leak-free pseudo-labelling, two stages per fold. v9 and v19 took their soft test labels from a fold-AVERAGED
submission, i.e. from models that had seen every validation fold — the OOF gain (+0.0002 for v9) showed on the
board as exactly +0.0000. Here fold b's soft labels come only from the stage-1 model trained on fold b's fit rows.
Stage 1 is also the clean best single: Naji-shaped LightGBM, nested TE m=1 on base + rounded keys, no pseudo.
Usage: python v22_pseudo_clean.py [params: naji|base|slow] [m] [fold seed] [TE seed]"""
import sys, numpy as np, pandas as pd, lightgbm as lgb, time
from sklearn.metrics import roc_auc_score
from features import *
pv = sys.argv[1] if len(sys.argv) > 1 else "naji"; m = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
fseed = int(sys.argv[3]) if len(sys.argv) > 3 else 42; tseed = int(sys.argv[4]) if len(sys.argv) > 4 else 0; tag = "" if (fseed, tseed) == (42, 0) else f"_f{fseed}_t{tseed}"
V = {"base": LGB_PARAMS, "slow": dict(LGB_PARAMS, learning_rate=0.02),
     "naji": dict(learning_rate=0.02, max_depth=5, num_leaves=32, min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_alpha=0.071, reg_lambda=2.0, max_bin=1024, verbose=-1)}
tr, te, o, y, feats = load(); base = Base(tr, te, o, feats); Btr, Bte = base(tr), base(te); skf = folds(tr, y, seed=fseed)
ref = roc_auc_score(y, np.load("submissions/oof_v6b_round.npy")); print(f"v6b ref OOF {ref:.6f}   params {pv}  m {m:g}  fold seed {fseed}  TE seed {tseed}", flush=True)
tenc = TE(tr, te, V5_SPECS + ROUND_SPECS, m=m, seed=tseed); p0 = {k: v for k, v in V[pv].items() if k != "n_estimators"} | dict(metric="auc")
p1, p2 = dict(p0, objective="binary"), dict(p0, objective="cross_entropy")
oof1 = np.zeros(len(tr)); oof2 = np.zeros(len(tr)); pred1 = np.zeros(len(te)); pred2 = np.zeros(len(te)); it1 = []; it2 = []; t = time.time()
for k, (a, b) in enumerate(skf):
    Xa, Xb, Xt = tenc.fold(a, b, y, Btr, Bte)
    d1 = lgb.Dataset(Xa, y[a]); v1 = lgb.Dataset(Xb, y[b], reference=d1)
    m1 = lgb.train(p1, d1, num_boost_round=20000, valid_sets=[v1], callbacks=[lgb.early_stopping(300, verbose=False)])
    oof1[b] = m1.predict(Xb); pt = m1.predict(Xt); pred1 += pt / 5; it1.append(m1.best_iteration)
    Xall = pd.concat([Xa, Xt], ignore_index=True); yall = np.r_[y[a], pt]
    d2 = lgb.Dataset(Xall, yall); v2 = lgb.Dataset(Xb, y[b], reference=d2)
    m2 = lgb.train(p2, d2, num_boost_round=20000, valid_sets=[v2], callbacks=[lgb.early_stopping(300, verbose=False)])
    oof2[b] = m2.predict(Xb); pred2 += m2.predict(Xt) / 5; it2.append(m2.best_iteration)
    print(f"  fold {k}: clean {roc_auc_score(y[b], oof1[b]):.6f} ({it1[-1]})  pseudo {roc_auc_score(y[b], oof2[b]):.6f} ({it2[-1]})  {time.time()-t:.0f}s", flush=True)
a1, a2 = roc_auc_score(y, oof1), roc_auc_score(y, oof2)
n1, n2 = f"v23_{pv}_m{m:g}{tag}", f"v22_pseudo_clean_{pv}_m{m:g}{tag}"
print(f"{n1} OOF {a1:.6f}  Δ vs v6b {a1-ref:+.6f}  (no pseudo, leak-free)\n{n2} OOF {a2:.6f}  Δ vs v6b {a2-ref:+.6f}  Δ vs clean {a2-a1:+.6f}  {time.time()-t:.0f}s", flush=True)
for n_, o_, p_ in [(n1, oof1, pred1), (n2, oof2, pred2)]:
    np.save(f"submissions/oof_{n_}.npy", o_); pd.DataFrame({"id": te.id, "Will_Buy_EV": p_}).to_csv(f"submissions/{n_}.csv", index=False)
