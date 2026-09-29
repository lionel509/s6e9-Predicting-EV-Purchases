"""refit_lgb: the heuljax refit/fold-average test mix (run_refit_ho.sh) for v37's LightGBM modes. Takes v37's own argv
(mode, n_splits, seed, init key) so v37_hybrid_modes' module-level settings and frame builders are reused verbatim.
  HOLDOUT=1  fold 0 of a 10-fold s7 split stands in for test: runs the N-fold loop on the rest, then refits on all of it
             at MULTS x mean best iteration, and prints holdout AUC of the fold average, each refit, and each 50/50 rank mix.
  otherwise  one full-train refit at MULT x that seed's own mean best iteration (from its json), written as a RANK column
             (v37's fold-average test columns are fold-averaged ranks, so the two mix on the same scale in assemble.py).
Usage: [HOLDOUT=1] [MULT=1.25] python refit_lgb.py agg 20 <seed> k_inc100   (issue #1)"""
import os, sys, time, json, numpy as np, pandas as pd, lightgbm as lgb
from scipy.special import logit, expit
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import v37_hybrid_modes as v
from features import load
from v27_hybrid import build, PARAMS, TARGET
assert v.MODE == "agg" and v.INIT, "only agg with an init key is wired up"
HOLDOUT = os.environ.get("HOLDOUT") == "1"; MULTS = [1.0, 1.25, 1.5]; MULT = float(os.environ.get("MULT", "1.25"))
rk = lambda p: rankdata(p) / len(p)
ini = lambda D: logit(np.clip(D[f"{v.INIT}_teauto"].to_numpy(float), 1e-4, 1 - 1e-4))

def refit(X, Xte, K, Kte, y, n_trees):
    a = np.arange(len(X)); A, _, C = v.frames(X, Xte, K, Kte, y, a, a[:5])   # dummy b, as in refit_full.py
    m = lgb.LGBMClassifier(random_state=v.SEED, **dict(PARAMS, n_estimators=n_trees)).fit(A, y, init_score=ini(A))
    return expit(m.predict(C, raw_score=True) + ini(C))

t = time.time(); tr, te, o, y, feats = load()
if v.SMOKE: tr, y, te = tr.iloc[:30000].reset_index(drop=True), y[:30000], te.iloc[:5000].reset_index(drop=True); PARAMS = dict(PARAMS, n_estimators=60)
if HOLDOUT:
    keep, ho = next(StratifiedKFold(10, shuffle=True, random_state=7).split(tr, y))
    te, y_ho = tr.iloc[ho].drop(columns=[TARGET], errors="ignore").reset_index(drop=True), y[ho]
    tr, y = tr.iloc[keep].reset_index(drop=True), y[keep]
X, Xte, K, Kte = build(tr, te, o); v.agg_frame(X, Xte, tr, te)
if HOLDOUT:
    pte, its = np.zeros(len(Xte)), []
    for f, (a, b) in enumerate(StratifiedKFold(v.N, shuffle=True, random_state=v.SEED).split(X, y)):
        A, B, C = v.frames(X, Xte, K, Kte, y, a, b)
        m = lgb.LGBMClassifier(random_state=v.SEED, **PARAMS)
        m.fit(A, y[a], init_score=ini(A), eval_set=[(B, y[b])], eval_init_score=[ini(B)], eval_metric="auc", callbacks=[lgb.early_stopping(500, verbose=False)])
        pte += rk(expit(m.predict(C, raw_score=True) + ini(C))) / v.N; its.append(m.best_iteration_)
        print(f"  fold {f}: trees {its[-1]}  {time.time()-t:.0f}s", flush=True)
    print(f"{v.name}_ho HOLDOUT AUC of the fold average {roc_auc_score(y_ho, pte):.6f}  mean trees {np.mean(its):.0f}", flush=True)
    for mult in MULTS:
        p = refit(X, Xte, K, Kte, y, int(round(np.mean(its) * mult)))
        print(f"  refit x{mult}: alone {roc_auc_score(y_ho, p):.6f}  50/50 rank mix {roc_auc_score(y_ho, pte + rk(p)):.6f}  {time.time()-t:.0f}s", flush=True)
else:
    n = int(round(np.mean(json.load(open(f"submissions/{v.name}.json"))["iters"]) * MULT))
    out = f"submissions/refit_{v.name}_m{MULT:g}.csv"
    pd.DataFrame({"id": te.id, TARGET: rk(refit(X, Xte, K, Kte, y, n))}).to_csv(out, index=False)
    print(f"wrote {out}  n_trees {n}  {time.time()-t:.0f}s", flush=True)
