"""How much of each family's OOF is early-stopping optimism? Re-run the 10-fold s42 hybrid and the init100 view with a FIXED tree
count (each model's own mean best_iteration from its ES run) and no eval set, and compare with the ES'd OOF. The public board fell
0.94646 → 0.94644 → 0.94643 as the init-score family's weight rose (blend_v18 → v20 → v23) while the OOF rose; if the init family's
ES optimism is larger than the hybrid's, that explains it and the final selection should hedge. Usage: python es_optimism.py"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "models"))   # models/ holds features + the v* scripts
import json, time, numpy as np, lightgbm as lgb
from scipy.special import logit, expit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from features import load
from v27_hybrid import build, fold_frames, PARAMS
tr, te, o, y, feats = load(); X, Xte, K, Kte = build(tr, te, o); t = time.time()
for name, key in (("v27_hybrid_k10_s42", None), ("v34_init_inc100_k10_s42", "k_inc100")):
    j = json.load(open(f"submissions/{name}.json")); n_fix = int(round(np.mean(j["iters"]))); es_auc = j["oof_auc"]
    prm = dict(PARAMS); prm["n_estimators"] = n_fix; oof = np.zeros(len(X))
    for f, (a, b) in enumerate(StratifiedKFold(10, shuffle=True, random_state=42).split(X, y)):
        A, B, C = fold_frames(X, Xte, K, Kte, y, a, b); m = lgb.LGBMClassifier(random_state=42, **prm)
        if key:
            ia, ib = [logit(np.clip(D[f"{key}_teauto"].to_numpy(float), 1e-4, 1 - 1e-4)) for D in (A, B)]
            m.fit(A, y[a], init_score=ia); oof[b] = expit(m.predict(B, raw_score=True) + ib)
        else:
            m.fit(A, y[a]); oof[b] = m.predict_proba(B)[:, 1]
        print(f"  {name} fold {f}: {roc_auc_score(y[b], oof[b]):.6f}  ({time.time()-t:.0f}s)", flush=True)
    auc = roc_auc_score(y, oof); print(f"{name}: ES OOF {es_auc:.6f}  fixed {n_fix} trees OOF {auc:.6f}  ES optimism {es_auc-auc:+.6f}", flush=True)
    np.save(f"submissions/oof_{name}_fixed{n_fix}.npy", oof)
