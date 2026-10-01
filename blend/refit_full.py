"""refit_full: full-train fixed-iteration refits, for the test column only (no OOF by construction — every fit row was
also used to fit, so an OOF here would be leaked). n_trees per seed is round(mean(best_iteration)) read off that seed's
own k10 early-stopped run, so the refit doesn't need (and can't have) an eval set of its own. fold_frames needs a
non-empty validation index, so we pass it a tiny dummy slice of the fit rows and discard its output (B is never used).
  init100  v34's k_inc100 init-score hybrid: fit rows get init_score = logit(nested k_inc100_teauto) exactly as
           training (fold_frames on ALL train rows as the fit set); test predicted as expit(raw + logit(test's
           whole-train k_inc100_teauto)). Reads submissions/v34_init_inc100_k10_s{seed}.json.
  hybrid   v27's hybrid, no init. Reads submissions/v27_hybrid_k10_s{seed}.json.
Usage: python refit_full.py <init100|hybrid> <seed> [seed ...]     SMOKE=1: 30k rows, n_trees capped at 60, no files written"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "models"))   # models/ holds features + the v* scripts
import os, sys, time, json, numpy as np, pandas as pd, lightgbm as lgb
from scipy.special import logit, expit
from features import load
from v27_hybrid import build, fold_frames, PARAMS, TARGET
MODEL = sys.argv[1]; SEEDS = [int(s) for s in sys.argv[2:]]
SMOKE = os.environ.get("SMOKE") == "1"
KEY = "k_inc100"
SRC = {"init100": "v34_init_inc100_k10_s{seed}", "hybrid": "v27_hybrid_k10_s{seed}"}[MODEL]

if __name__ == "__main__":
    t = time.time(); tr, te, o, y, feats = load()
    if SMOKE: tr = tr.iloc[:30000].reset_index(drop=True); y = y[:30000]; te = te.iloc[:5000].reset_index(drop=True)
    X, Xte, K, Kte = build(tr, te, o)
    a = np.arange(len(X)); b = a[:5]   # dummy: fold_frames can't take an empty validation index; B is unused below
    # the nested encoding doesn't depend on seed (fold_frames' inner TargetEncoder is always random_state=42, like
    # v27/v34's own fold loops), so build it once and reuse it for every seed below
    A, _, C = fold_frames(X, Xte, K, Kte, y, a, b)
    print(f"refit_full {MODEL}: {len(a)} fit rows, {len(Xte)} test rows, seeds {SEEDS}  ({time.time()-t:.0f}s)", flush=True)
    for seed in SEEDS:
        jpath = f"submissions/{SRC.format(seed=seed)}.json"
        j = json.load(open(jpath)); n_trees = int(round(np.mean(j["iters"])))
        if SMOKE: n_trees = min(n_trees, 60)
        prm = dict(PARAMS); prm["n_estimators"] = n_trees
        m = lgb.LGBMClassifier(random_state=seed, **prm)
        if MODEL == "init100":
            ia = logit(np.clip(A[f"{KEY}_teauto"].to_numpy(float), 1e-4, 1 - 1e-4))
            ic = logit(np.clip(C[f"{KEY}_teauto"].to_numpy(float), 1e-4, 1 - 1e-4))
            m.fit(A, y[a], init_score=ia)
            pc = expit(m.predict(C, raw_score=True) + ic)
        else:
            m.fit(A, y[a])
            pc = m.predict_proba(C)[:, 1]
        print(f"  seed {seed}: n_trees {n_trees} (from {jpath})  fit done  {time.time()-t:.0f}s", flush=True)
        if SMOKE: continue
        out = f"submissions/refit_{MODEL}_s{seed}.csv"
        pd.DataFrame({"id": te.id, TARGET: pc}).to_csv(out, index=False)
        print(f"  wrote {out}  {time.time()-t:.0f}s", flush=True)
