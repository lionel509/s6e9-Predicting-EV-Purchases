"""v38: v34's init-score model with a LightGBM *shape* flag — single-lever parameter variants of the strongest own member, so the
init view can be probed for capacity / learning-rate headroom without touching the frame, the init logits or the fold loop.
FLAG (4th arg) selects the patch: slow = learning_rate 0.01 with the n_estimators cap doubled (same budget in smaller steps),
wide = num_leaves 64 / max_depth 6 / min_child_samples 20, col5 = colsample_bytree 0.5. drop and xgb keep v34's exact meaning
(the key's TE columns removed / megayak's view-B XGBoost learner, no shape patch); no FLAG runs plain v34 params. Output name
v38_init_<key>_p<FLAG>_k<N>_s<SEED>, e.g. v38_init_inc100_pslow_k10_s42 (the _p<FLAG> part is omitted when FLAG is omitted).
The JSON sidecar keeps the per-fold 'iters' list refit_full.py reads.
Usage: python v38_init_shapes.py [n_folds=5] [seed=42] [key=k_inc_exact] [slow|wide|col5|drop|xgb]     SMOKE=1: 30k rows, 60 trees"""
import os, sys, time, json, numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from scipy.special import logit, expit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from features import load
from v27_hybrid import build, fold_frames, PARAMS, TARGET
rk = lambda v: rankdata(v) / len(v)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5; SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42
KEY = sys.argv[3] if len(sys.argv) > 3 else "k_inc_exact"; FLAG = sys.argv[4] if len(sys.argv) > 4 else None
FLAGS = ("slow", "wide", "col5", "drop", "xgb")
if FLAG is not None and FLAG not in FLAGS: sys.exit(f"usage: v38_init_shapes.py [folds] [seed] [key] [{'|'.join(FLAGS)}]")
DROP = FLAG == "drop"; XGB = FLAG == "xgb"; SMOKE = os.environ.get("SMOKE") == "1"
SHAPE = {"slow": dict(learning_rate=0.01), "wide": dict(num_leaves=64, max_depth=6, min_child_samples=20),
         "col5": dict(colsample_bytree=0.5)}.get(FLAG)
KEYS = KEY.split("+")
name = f"v38_init_{KEY.replace('k_', '').replace('+', 'p')}{f'_p{FLAG}' if FLAG else ''}_k{N}_s{SEED}" + ("_smoke" if SMOKE else "")
XGB_PARAMS = dict(n_estimators=20000, learning_rate=0.02, max_depth=5, min_child_weight=5, subsample=0.8, colsample_bytree=0.3, reg_alpha=0.071,
                   reg_lambda=2.0, max_bin=1024, tree_method="hist", enable_categorical=True, eval_metric="auc", early_stopping_rounds=500,
                   n_jobs=-1, verbosity=0)
if __name__ == "__main__":
    t = time.time(); tr, te, o, y, feats = load()
    if SMOKE: tr = tr.iloc[:30000].reset_index(drop=True); y = y[:30000]; te = te.iloc[:5000].reset_index(drop=True)
    X, Xte, K, Kte = build(tr, te, o)
    refp = f"submissions/oof_v27_hybrid_k{N}_s{SEED}.npy"; ref = roc_auc_score(y, np.load(refp)) if os.path.exists(refp) and not SMOKE else float("nan")
    print(f"{name}: v27 ref OOF {ref:.6f}  features {X.shape[1]}  init from {'+'.join(KEYS)}_teauto{' (its TE columns dropped)' if DROP else ''}{'  learner XGB' if XGB else ''}", flush=True)
    cv = StratifiedKFold(N, shuffle=True, random_state=SEED); oof = np.zeros(len(X)); oof_rk = np.zeros(len(X)); pte = np.zeros(len(Xte)); its = []
    prm = dict(PARAMS); prm.update(SHAPE) if SHAPE else None
    if FLAG == "slow": prm["n_estimators"] *= 2   # half the step size, same budget in trees
    print(f"  lgb params ({FLAG or 'base'}): {prm}", flush=True)
    prm.update(n_estimators=60) if SMOKE else None
    xprm = dict(XGB_PARAMS, random_state=SEED); xprm.update(n_estimators=60) if SMOKE else None
    for f, (a, b) in enumerate(cv.split(X, y)):
        A, B, C = fold_frames(X, Xte, K, Kte, y, a, b)
        ini = [np.mean([logit(np.clip(D[f"{k}_teauto"].to_numpy(float), 1e-4, 1 - 1e-4)) for k in KEYS], axis=0) for D in (A, B, C)]
        if DROP:
            cols = [c for c in A.columns if any(c.startswith(f"{k}_te") for k in KEYS)]; A, B, C = A.drop(columns=cols), B.drop(columns=cols), C.drop(columns=cols)
        if XGB:
            m = xgb.XGBClassifier(**xprm)
            m.fit(A, y[a], base_margin=ini[0], eval_set=[(B, y[b])], base_margin_eval_set=[ini[1]], verbose=False)
            pb = expit(m.predict(B, output_margin=True, base_margin=ini[1])); pc = expit(m.predict(C, output_margin=True, base_margin=ini[2]))
            its.append(int(m.best_iteration))
        else:
            m = lgb.LGBMClassifier(random_state=SEED, **prm)
            m.fit(A, y[a], init_score=ini[0], eval_set=[(B, y[b])], eval_init_score=[ini[1]], eval_metric="auc", callbacks=[lgb.early_stopping(500, verbose=False)])
            pb = expit(m.predict(B, raw_score=True) + ini[1]); pc = expit(m.predict(C, raw_score=True) + ini[2]); its.append(m.best_iteration_)
        oof[b] = pb; oof_rk[b] = rk(pb); pte += rk(pc) / N
        print(f"  fold {f}: auc {roc_auc_score(y[b], pb):.6f}  (init alone {roc_auc_score(y[b], ini[1]):.6f})  trees {its[-1]}  {time.time()-t:.0f}s", flush=True)
    auc, auc_rk = roc_auc_score(y, oof), roc_auc_score(y, oof_rk)
    print(f"{name} OOF {auc:.6f}  (fold-ranked {auc_rk:.6f})  Δ vs v27 {auc-ref:+.6f}  iters {its}  {time.time()-t:.0f}s", flush=True)
    if SMOKE: sys.exit(0)
    np.save(f"submissions/oof_{name}.npy", oof); pd.DataFrame({"id": te.id, TARGET: pte}).to_csv(f"submissions/{name}.csv", index=False)
    json.dump({"oof_auc": auc, "ref_v27": ref, "iters": its, "n_splits": N, "seed": SEED, "key": KEY, "drop": DROP, "xgb": XGB, "flag": FLAG}, open(f"submissions/{name}.json", "w"), indent=2)
