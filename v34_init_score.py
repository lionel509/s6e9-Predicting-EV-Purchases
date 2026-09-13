"""v34: the "init_score view" — megayak's suggested sixth member (the forum's 'everything we measured' thread called it the most
decorrelated member anyone built). Same hybrid frame and params as v27, but boosting starts from logit(smoothed per-value income
target rate) — the nested k_inc_exact TargetEncoder(smooth=auto) value — instead of from the base rate, so the trees learn the
correction to the encoding rather than re-deriving it. With DROP the key's own three TE columns are removed from the frame.
Usage: python v34_init_score.py [n_folds=5] [seed=42] [key=k_inc_exact] [drop]     SMOKE=1: 30k rows, 60 trees"""
import os, sys, time, json, numpy as np, pandas as pd, lightgbm as lgb
from scipy.special import logit, expit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from features import load
from v27_hybrid import build, fold_frames, PARAMS, TARGET
rk = lambda v: rankdata(v) / len(v)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5; SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42
KEY = sys.argv[3] if len(sys.argv) > 3 else "k_inc_exact"; DROP = len(sys.argv) > 4 and sys.argv[4] == "drop"; SMOKE = os.environ.get("SMOKE") == "1"
name = f"v34_init_{KEY.replace('k_', '')}{'_drop' if DROP else ''}_k{N}_s{SEED}" + ("_smoke" if SMOKE else "")
if __name__ == "__main__":
    t = time.time(); tr, te, o, y, feats = load()
    if SMOKE: tr = tr.iloc[:30000].reset_index(drop=True); y = y[:30000]; te = te.iloc[:5000].reset_index(drop=True)
    X, Xte, K, Kte = build(tr, te, o)
    refp = f"submissions/oof_v27_hybrid_k{N}_s{SEED}.npy"; ref = roc_auc_score(y, np.load(refp)) if os.path.exists(refp) and not SMOKE else float("nan")
    print(f"{name}: v27 ref OOF {ref:.6f}  features {X.shape[1]}  init from {KEY}_teauto{' (its TE columns dropped)' if DROP else ''}", flush=True)
    cv = StratifiedKFold(N, shuffle=True, random_state=SEED); oof = np.zeros(len(X)); oof_rk = np.zeros(len(X)); pte = np.zeros(len(Xte)); its = []
    prm = dict(PARAMS); prm.update(n_estimators=60) if SMOKE else None
    for f, (a, b) in enumerate(cv.split(X, y)):
        A, B, C = fold_frames(X, Xte, K, Kte, y, a, b)
        ini = [logit(np.clip(D[f"{KEY}_teauto"].to_numpy(float), 1e-4, 1 - 1e-4)) for D in (A, B, C)]
        if DROP:
            cols = [c for c in A.columns if c.startswith(f"{KEY}_te")]; A, B, C = A.drop(columns=cols), B.drop(columns=cols), C.drop(columns=cols)
        m = lgb.LGBMClassifier(random_state=SEED, **prm)
        m.fit(A, y[a], init_score=ini[0], eval_set=[(B, y[b])], eval_init_score=[ini[1]], eval_metric="auc", callbacks=[lgb.early_stopping(500, verbose=False)])
        pb = expit(m.predict(B, raw_score=True) + ini[1]); pc = expit(m.predict(C, raw_score=True) + ini[2]); its.append(m.best_iteration_)
        oof[b] = pb; oof_rk[b] = rk(pb); pte += rk(pc) / N
        print(f"  fold {f}: auc {roc_auc_score(y[b], pb):.6f}  (init alone {roc_auc_score(y[b], ini[1]):.6f})  trees {its[-1]}  {time.time()-t:.0f}s", flush=True)
    auc, auc_rk = roc_auc_score(y, oof), roc_auc_score(y, oof_rk)
    print(f"{name} OOF {auc:.6f}  (fold-ranked {auc_rk:.6f})  Δ vs v27 {auc-ref:+.6f}  iters {its}  {time.time()-t:.0f}s", flush=True)
    if SMOKE: sys.exit(0)
    np.save(f"submissions/oof_{name}.npy", oof); pd.DataFrame({"id": te.id, TARGET: pte}).to_csv(f"submissions/{name}.csv", index=False)
    json.dump({"oof_auc": auc, "ref_v27": ref, "iters": its, "n_splits": N, "seed": SEED, "key": KEY, "drop": DROP}, open(f"submissions/{name}.json", "w"), indent=2)
