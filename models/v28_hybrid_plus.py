"""v28: our own levers on top of the v27 hybrid frame, paired against v27 on the same split.
  --m1      add our nested TE (m=1, features.TE) of income / commute / age + income rounded 100 / 1000 — a fourth,
            nearly-unsmoothed encoding next to sklearn's auto / 10 / 100
  --pseudo  leak-free two-stage pseudo-labelling (v22's form): fold b's soft test labels come only from the stage-1
            model fitted on fold b's fit rows; stage 2 refits on fit rows + test rows with LightGBM cross_entropy
Stage 1 is always saved too. Usage: python v28_hybrid_plus.py [--m1] [--pseudo] [folds=5] [seed=42]"""
import sys, time, json, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from features import load, TE, V5_SPECS, ROUND_SPECS
from v27_hybrid import build, PARAMS, TARGET
flags = [a for a in sys.argv[1:] if a.startswith("--")]; pos = [a for a in sys.argv[1:] if not a.startswith("--")]
M1, PSEUDO = "--m1" in flags, "--pseudo" in flags
N = int(pos[0]) if pos else 5; SEED = int(pos[1]) if len(pos) > 1 else 42
base_name = f"v28_hybrid{'_m1' if M1 else ''}_k{N}_s{SEED}"; rk = lambda v: rankdata(v) / len(v)
t = time.time(); tr, te, o, y, feats = load(); X, Xte, K, Kte = build(tr, te, o)
ref = roc_auc_score(y, np.load("submissions/oof_v27_hybrid_k5_s42.npy")) if N == 5 and SEED == 42 else float("nan")
p = {k: v for k, v in PARAMS.items() if k not in ("n_estimators", "n_jobs")} | dict(num_threads=12, seed=SEED, metric="auc")
p1, p2 = dict(p, objective="binary"), dict(p, objective="cross_entropy")
tenc = TE(tr, te, V5_SPECS + ROUND_SPECS, m=1, seed=0) if M1 else None
print(f"{base_name}: v27 ref OOF {ref:.6f}  m1 {M1}  pseudo {PSEUDO}  ({time.time()-t:.0f}s)", flush=True)
cv = StratifiedKFold(N, shuffle=True, random_state=SEED)
oof1 = np.zeros(len(X)); oof2 = np.zeros(len(X)); pte1 = np.zeros(len(Xte)); pte2 = np.zeros(len(Xte)); it1 = []; it2 = []
for f, (a, b) in enumerate(cv.split(X, y)):
    if M1: A, B, C = tenc.fold(a, b, y, X, Xte)
    else: A, B, C = X.iloc[a].copy(), X.iloc[b].copy(), Xte.copy()
    for smooth, tg in (("auto", "auto"), (10.0, "10"), (100.0, "100")):
        enc = TargetEncoder(shuffle=True, cv=5, smooth=smooth, random_state=42)
        ea = enc.fit_transform(K.iloc[a], y[a]); eb, ec = enc.transform(K.iloc[b]), enc.transform(Kte)
        for i, c in enumerate(K.columns):
            A[f"{c}_te{tg}"] = ea[:, i].astype("float32"); B[f"{c}_te{tg}"] = eb[:, i].astype("float32"); C[f"{c}_te{tg}"] = ec[:, i].astype("float32")
    d1 = lgb.Dataset(A, y[a]); v1 = lgb.Dataset(B, y[b], reference=d1)
    m1 = lgb.train(p1, d1, num_boost_round=20000, valid_sets=[v1], callbacks=[lgb.early_stopping(500, verbose=False)])
    pb = m1.predict(B); pt = m1.predict(C); oof1[b] = pb; pte1 += rk(pt) / N; it1.append(m1.best_iteration)
    line = f"  fold {f}: stage1 {roc_auc_score(y[b], pb):.6f} ({it1[-1]})"
    if PSEUDO:
        Xall = pd.concat([A, C], ignore_index=True); yall = np.r_[y[a], pt]
        d2 = lgb.Dataset(Xall, yall); v2 = lgb.Dataset(B, y[b], reference=d2)
        m2 = lgb.train(p2, d2, num_boost_round=20000, valid_sets=[v2], callbacks=[lgb.early_stopping(500, verbose=False)])
        oof2[b] = m2.predict(B); pte2 += rk(m2.predict(C)) / N; it2.append(m2.best_iteration)
        line += f"  pseudo {roc_auc_score(y[b], oof2[b]):.6f} ({it2[-1]})"
    print(line + f"  {time.time()-t:.0f}s", flush=True)
a1 = roc_auc_score(y, oof1); print(f"{base_name} OOF {a1:.6f}  Δ vs v27 {a1-ref:+.6f}  iters {it1}  {time.time()-t:.0f}s", flush=True)
np.save(f"submissions/oof_{base_name}.npy", oof1); pd.DataFrame({"id": te.id, TARGET: pte1}).to_csv(f"submissions/{base_name}.csv", index=False)
if PSEUDO:
    n2 = base_name.replace("v28_hybrid", "v28_hybrid_pseudo"); a2 = roc_auc_score(y, oof2)
    print(f"{n2} OOF {a2:.6f}  Δ vs v27 {a2-ref:+.6f}  Δ vs stage1 {a2-a1:+.6f}  iters {it2}", flush=True)
    np.save(f"submissions/oof_{n2}.npy", oof2); pd.DataFrame({"id": te.id, TARGET: pte2}).to_csv(f"submissions/{n2}.csv", index=False)
