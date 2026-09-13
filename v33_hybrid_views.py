"""v33: megayak's feature views C and D ported onto our pipeline (so we can seed-bag them), plus his suggested fifth member E.
  C  no digit block; lift/novelty vs the original; centred-window target rates (income ±2..±200, commute ±1..±10, income ±5/±25
     within city|car); TE smoothing 2/30/300
  D  digit block kept; NO exact-income key: //10, //50, //500, //5000 income keys + //5, //50 commute keys; windows; smoothing auto/20/200
  E  (ours) D with the //25, //250, //2500 ladder instead — the "obvious fifth member" from megayak's write-up
  F  (ours) C × D: no digit block, no exact key, //10/50/500/5000 ladder, windows, lift; smoothing 2/30/300 — furthest from A
Frozen partition StratifiedKFold(N, shuffle, seed) like v27; TE and window rates are nested (inner 5-fold for the fit rows, full fit
set for validation and test). OOF saved as raw probabilities like every oof_*.npy here (megayak's files are fold-ranked).
Reference on the same 10-fold s42 split: megayak C 0.946223, D 0.946077, A 0.946264.
Usage: python v33_hybrid_views.py VIEW [n_folds=5] [seed=42] [lgb|xgb] [init key]
  "xgb" swaps in megayak's view-B XGBoost (hist, depth 5, colsample 0.3, max_bin 1024, ES 500); an init key (e.g. k_inc50) makes the
  LightGBM run start from logit of that key's nested auto-smoothed target rate (the v34 idea on this frame)"""
import sys, time, json, numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from scipy.special import logit, expit
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from features import load
from v27_hybrid import PARAMS, TARGET, CATS, NUMS
rk = lambda v: rankdata(v) / len(v)
VIEW = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 5; SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 42
XGB = len(sys.argv) > 4 and sys.argv[4] == "xgb"; INIT = sys.argv[5] if len(sys.argv) > 5 else None
name = f"v33_view{VIEW}{'_xgb' if XGB else ''}{'_init' + INIT.replace('k_', '') if INIT else ''}_k{N}_s{SEED}"
CFG = {"C": dict(digits=False, lift=True, keys="default", smooth=(2.0, 30.0, 300.0)),
       "D": dict(digits=True, lift=False, keys="alt", smooth=("auto", 20.0, 200.0)),
       "E": dict(digits=True, lift=False, keys="alt25", smooth=("auto", 20.0, 200.0)),
       "F": dict(digits=False, lift=True, keys="alt", smooth=(2.0, 30.0, 300.0))}[VIEW]

def build(train, test, orig, digits=True, lift=False, keys="default"):
    n = len(train)
    df = pd.concat([train[CATS + NUMS], test[CATS + NUMS]], ignore_index=True)
    inc = df.Annual_Income_USD.to_numpy(np.int64); km = np.round(df.Daily_Commute_km.to_numpy(float) * 10).astype(np.int64)
    if digits:
        df["inc_d1"] = (inc % 10).astype("int8"); df["inc_d2"] = (inc // 10 % 10).astype("int8"); df["inc_d3"] = (inc // 100 % 10).astype("int8")
        df["inc_mod100"] = (inc % 100).astype("int16"); df["inc_mod1000"] = (inc % 1000).astype("int16")
        df["km_d1"] = (km % 10).astype("int8"); df["km_mod100"] = (km % 100).astype("int8")
        for d in (50, 100, 250, 500, 1000, 2500, 5000): df[f"inc_q{d}"] = (inc // d).astype("int32")
        for d in (5, 10, 25, 50): df[f"km_q{d}"] = (km // d).astype("int32")
        s = pd.Series(inc); df["fq_inc"] = s.map(s.value_counts()).astype("float32").values
        s = pd.Series(km); df["fq_km"] = s.map(s.value_counts()).astype("float32").values
    df["is_30k_spike"] = (inc == 30000).astype("int8"); df["is_millionaire_cliff"] = (inc >= 170537).astype("int8")
    df["is_dead_zone"] = ((inc >= 38000) & (inc <= 42000)).astype("int8"); df["is_env_hater"] = (df.Environmental_Concern_Level == 1).astype("int8")
    og = orig.dropna(subset=["Annual_Income_USD", "Daily_Commute_km", "Environmental_Concern_Level"]).copy()
    og[TARGET] = (og[TARGET] == "Yes").astype(int); gm = og[TARGET].mean()
    for c in CATS + NUMS: df[f"{c}_org_mean"] = df[c].map(og.groupby(c)[TARGET].mean()).fillna(gm).astype("float32")
    if lift:   # how much the generator over-produced each value vs the original
        oi = np.round(orig.Annual_Income_USD.dropna().to_numpy(float)).astype(np.int64); ok = np.round(orig.Daily_Commute_km.dropna().to_numpy(float) * 10).astype(np.int64)
        for nm, comp, src, divs in (("inc", inc, oi, (1, 100, 1000)), ("km", km, ok, (1, 10))):
            for d in divs:
                c_s = pd.Series(comp // d); o_s = pd.Series(src // d)
                c_cnt = c_s.map(c_s.value_counts()).to_numpy(float); o_cnt = c_s.map(o_s.value_counts()).fillna(0).to_numpy(float)
                sfx = "" if d == 1 else f"_q{d}"
                df[f"lift_{nm}{sfx}"] = (np.log((c_cnt + 1) / len(df)) - np.log((o_cnt + 1) / len(src))).astype("float32"); df[f"ocnt_{nm}{sfx}"] = o_cnt.astype("float32")
                if d == 1: df[f"novel_{nm}"] = (o_cnt == 0).astype("int8")
    te = {}
    if keys == "alt":
        for d in (10, 50, 500, 5000): te[f"k_inc{d}"] = (inc // d).astype(str)
        for d in (5, 50): te[f"k_km{d}"] = (km // d).astype(str)
    elif keys == "alt25":
        for d in (25, 250, 2500): te[f"k_inc{d}"] = (inc // d).astype(str)
        for d in (5, 50): te[f"k_km{d}"] = (km // d).astype(str)
    else:
        te["k_inc_exact"] = inc.astype(str); te["k_inc100"] = (inc // 100).astype(str); te["k_inc1000"] = (inc // 1000).astype(str); te["k_km_int"] = (km // 10).astype(str)
    for c in CATS + ["Age", "Number_of_Cars_Owned", "Charging_Stations_Near_Home", "Charging_Stations_Near_Work", "Environmental_Concern_Level"]:
        te[f"k_{c}"] = df[c].astype(str).to_numpy()
    K = pd.DataFrame(te)
    for c in K.columns: df[f"{c}_fe"] = K[c].map(K[c].value_counts(normalize=True)).astype("float32").values
    for c in CATS: df[c] = df[c].astype("category")
    cut = lambda x: (x.iloc[:n].reset_index(drop=True), x.iloc[n:].reset_index(drop=True))
    (X, Xte), (Kt, Kte) = cut(df), cut(K)
    return X, Xte, Kt, Kte

WIN_INC, WIN_KM, WIN_GRP, WIN_PRIOR = (2, 5, 10, 25, 50, 200), (1, 3, 10), (5, 25), 10.0
WIN_COLS = [f"win_inc_{w}" for w in WIN_INC] + [f"win_km_{w}" for w in WIN_KM] + [f"win_inc_{w}_citycar" for w in WIN_GRP]
def window_rates(v_fit, y_fit, v_query, widths, prior, gm):
    """Smoothed buy rate over the centred window [v-w, v+w], from (v_fit, y_fit) only."""
    lo, hi = int(min(v_fit.min(), v_query.min())), int(max(v_fit.max(), v_query.max())); n = hi - lo + 1
    cnt = np.bincount(v_fit - lo, minlength=n).astype(np.float64); s = np.bincount(v_fit - lo, weights=y_fit, minlength=n)
    ccnt = np.concatenate([[0.0], np.cumsum(cnt)]); cs = np.concatenate([[0.0], np.cumsum(s)]); q = v_query - lo; out = []
    for w in widths:
        a = np.clip(q - w, 0, n); b = np.clip(q + w + 1, 0, n)
        out.append((((cs[b] - cs[a]) + prior * gm) / ((ccnt[b] - ccnt[a]) + prior)).astype("float32"))
    return np.stack(out, 1)
def window_block(inc_a, km_a, y_a, inc_q, km_q, g_a, g_q):
    gm = float(y_a.mean())
    parts = [window_rates(inc_a, y_a, inc_q, WIN_INC, WIN_PRIOR, gm), window_rates(km_a, y_a, km_q, WIN_KM, WIN_PRIOR, gm)]
    out = np.full((len(inc_q), len(WIN_GRP)), gm, np.float32)
    for g in np.unique(g_q):
        ma, mq = g_a == g, g_q == g
        if ma.sum(): out[mq] = window_rates(inc_a[ma], y_a[ma], inc_q[mq], WIN_GRP, WIN_PRIOR, gm)
    parts.append(out); return np.concatenate(parts, 1)

if __name__ == "__main__":
    t = time.time(); tr, te, o, y, feats = load(); X, Xte, K, Kte = build(tr, te, o, digits=CFG["digits"], lift=CFG["lift"], keys=CFG["keys"])
    n = len(X); inc_tr, inc_te = X.Annual_Income_USD.to_numpy(np.int64), Xte.Annual_Income_USD.to_numpy(np.int64)
    km_tr, km_te = np.round(X.Daily_Commute_km.to_numpy(float) * 10).astype(np.int64), np.round(Xte.Daily_Commute_km.to_numpy(float) * 10).astype(np.int64)
    g_all = pd.factorize(pd.concat([X.City_Type.astype(str) + "|" + X.Current_Car_Type.astype(str), Xte.City_Type.astype(str) + "|" + Xte.Current_Car_Type.astype(str)], ignore_index=True))[0]
    g_tr, g_te = g_all[:n], g_all[n:]
    print(f"{name}: base features {X.shape[1]} | TE keys {K.shape[1]} x {CFG['smooth']} | windows {len(WIN_COLS)}  ({time.time()-t:.0f}s)", flush=True)
    cv = StratifiedKFold(N, shuffle=True, random_state=SEED); oof = np.zeros(n); oof_rk = np.zeros(n); pte = np.zeros(len(Xte)); iters = []
    for f, (a, b) in enumerate(cv.split(X, y)):
        A, B, C = X.iloc[a].copy(), X.iloc[b].copy(), Xte.copy()
        for sm in CFG["smooth"]:
            tag = "auto" if sm == "auto" else str(int(sm)); enc = TargetEncoder(shuffle=True, cv=5, smooth=sm, random_state=42)
            ea = enc.fit_transform(K.iloc[a], y[a]); eb, ec = enc.transform(K.iloc[b]), enc.transform(Kte)
            for i, c in enumerate(K.columns):
                A[f"{c}_te{tag}"] = ea[:, i].astype("float32"); B[f"{c}_te{tag}"] = eb[:, i].astype("float32"); C[f"{c}_te{tag}"] = ec[:, i].astype("float32")
        wa = np.zeros((len(a), len(WIN_COLS)), np.float32)
        for ia, ib in StratifiedKFold(5, shuffle=True, random_state=42).split(a, y[a]):
            wa[ib] = window_block(inc_tr[a[ia]], km_tr[a[ia]], y[a[ia]].astype(float), inc_tr[a[ib]], km_tr[a[ib]], g_tr[a[ia]], g_tr[a[ib]])
        wb = window_block(inc_tr[a], km_tr[a], y[a].astype(float), inc_tr[b], km_tr[b], g_tr[a], g_tr[b])
        wc = window_block(inc_tr[a], km_tr[a], y[a].astype(float), inc_te, km_te, g_tr[a], g_te)
        for i, c in enumerate(WIN_COLS): A[c] = wa[:, i]; B[c] = wb[:, i]; C[c] = wc[:, i]
        if XGB:
            m = xgb.XGBClassifier(n_estimators=20000, learning_rate=0.02, max_depth=5, min_child_weight=5, subsample=0.8, colsample_bytree=0.3, reg_alpha=0.071,
                                  reg_lambda=2.0, max_bin=1024, tree_method="hist", enable_categorical=True, eval_metric="auc", early_stopping_rounds=500,
                                  random_state=SEED, n_jobs=-1, verbosity=0)
            m.fit(A, y[a], eval_set=[(B, y[b])], verbose=False); it = int(m.best_iteration)
        elif INIT:
            ini = [logit(np.clip(D[f"{INIT}_teauto"].to_numpy(float), 1e-4, 1 - 1e-4)) for D in (A, B, C)]
            m = lgb.LGBMClassifier(random_state=SEED, **PARAMS)
            m.fit(A, y[a], init_score=ini[0], eval_set=[(B, y[b])], eval_init_score=[ini[1]], eval_metric="auc", callbacks=[lgb.early_stopping(500, verbose=False)]); it = int(m.best_iteration_)
        else:
            m = lgb.LGBMClassifier(random_state=SEED, **PARAMS)
            m.fit(A, y[a], eval_set=[(B, y[b])], eval_metric="auc", callbacks=[lgb.early_stopping(500, verbose=False)]); it = int(m.best_iteration_)
        if INIT and not XGB: pb = expit(m.predict(B, raw_score=True) + ini[1]); pc = expit(m.predict(C, raw_score=True) + ini[2])
        else: pb = m.predict_proba(B)[:, 1]; pc = m.predict_proba(C)[:, 1]
        oof[b] = pb; oof_rk[b] = rk(pb); pte += rk(pc) / N; iters.append(it)
        print(f"  fold {f}: auc {roc_auc_score(y[b], pb):.6f}  trees {it}  ncol {A.shape[1]}  {time.time()-t:.0f}s", flush=True)
    auc, auc_rk = roc_auc_score(y, oof), roc_auc_score(y, oof_rk)
    print(f"{name} OOF {auc:.6f}  (fold-ranked {auc_rk:.6f})  iters {iters}  {time.time()-t:.0f}s", flush=True)
    np.save(f"submissions/oof_{name}.npy", oof); pd.DataFrame({"id": te.id, TARGET: pte}).to_csv(f"submissions/{name}.csv", index=False)
    json.dump({"oof_auc": auc, "oof_auc_fold_ranked": auc_rk, "iters": iters, "n_splits": N, "seed": SEED, "view": VIEW}, open(f"submissions/{name}.json", "w"), indent=2)
