"""v39: heuljax's stacking lever (public rank 5, 0.94674, 24 subs — "predictions [target & non-target] from inner-fold
basic models as features") on our strongest formulation: v34's init100 LightGBM with inner-fold basic-model predictions
added as columns. Per outer fold the FIT rows get cross-fitted basic predictions — a 5-fold inner StratifiedKFold
(seed SEED+1000+f) where each basic model trains on fold_frames(d, r) DONOR-ONLY target encodings, never on A — and
validation/test get predictions from the basic models refit on all outer fit rows, so no stacked value ever saw its own
label. Two basic models: rawlgb (raw CATS+NUMS+freqs+flags+two original means, one shallow fast LightGBM, no eval_set)
and lr (LogisticRegression over every _te column, NUMS, log1p freqs, the 4 flags and one-hot CATS, non-dummies
standardised with the DONOR's mean/std). Stack columns added to A/B/C (float32): stk_rawlgb / stk_lr logits (p clipped
to [1e-5, 1-1e-5]), the exact / inc100 surprisal of the rawlgb logit against the frame's OWN nested TE, and
stk_lr_minus_raw. VARIANT picks the block: t = target stack only, n = non-target block only (v37's label-free
agg_inc_* / agg_inc100_* income aggregates over train+test plus dev_env / dev_sub / dev_rah row deviations, built once
before the fold loop — no inner loop), tn = both, tn_initlr = both with the main booster started from the cross-fitted
LR logit instead of logit(k_inc100_teauto). Everything else is v34 verbatim: triple TE fold_frames, Naji-shaped params,
ES 500, per-fold rank-averaged test column, oof_*.npy + .csv + .json outputs.
Usage: python v39_stacked_features.py VARIANT [n_folds=10] [seed=42]    VARIANT in t | n | tn | tn_initlr
       SMOKE=1: 30k train / 5k test, main 60 trees, rawlgb 30 trees, LR 20 iters (inner folds stay 5)"""
import os, sys, time, json, gc, numpy as np, pandas as pd, lightgbm as lgb
from scipy.special import logit, expit
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from features import load
from v27_hybrid import build, fold_frames, PARAMS, TARGET, CATS, NUMS
rk = lambda v: rankdata(v) / len(v)
VARIANT = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 10; SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 42
assert VARIANT in ("t", "n", "tn", "tn_initlr"), f"VARIANT must be t|n|tn|tn_initlr, got {VARIANT!r}"
DO_STACK, DO_AGG, INIT_LR = VARIANT != "n", VARIANT != "t", VARIANT == "tn_initlr"
SMOKE = os.environ.get("SMOKE") == "1"
name = f"v39_stack_{VARIANT}_k{N}_s{SEED}" + ("_smoke" if SMOKE else "")
KEYS = ["k_inc100"]
FLAG4 = ["is_30k_spike", "is_millionaire_cliff", "is_dead_zone", "is_env_hater"]
RAWLGB = CATS + NUMS + ["fq_inc", "fq_km"] + FLAG4 + ["Annual_Income_USD_org_mean", "Daily_Commute_km_org_mean"]
STACK_COLS = ["stk_rawlgb", "stk_lr", "stk_surp_exact", "stk_surp_100", "stk_lr_minus_raw"]
RAWLGB_PARAMS = dict(n_estimators=30 if SMOKE else 300, learning_rate=0.1, num_leaves=31, min_child_samples=100, max_bin=63,
                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, verbose=-1, random_state=SEED)

def agg_frame(X, Xte, tr, te):
    """(v37 verbatim) Label-free income aggregates over the raw train+test rows, mapped on as agg_inc_* / agg_inc100_*
    float32 columns. env/sub/rah/ral/km/age/home = mean concern, share subsidy Yes, share range High, share range Low,
    mean commute, mean age, share home charging Yes. No labels used (the frame is built before any y enters), so no fold
    nesting needed."""
    raw = pd.concat([tr, te], ignore_index=True)
    d = pd.DataFrame({"env": raw.Environmental_Concern_Level, "km": raw.Daily_Commute_km, "age": raw.Age,
                      "sub": (raw.Subsidy_Available == "Yes").astype("float64"), "rah": (raw.Range_Anxiety_Level == "High").astype("float64"),
                      "ral": (raw.Range_Anxiety_Level == "Low").astype("float64"), "home": (raw.Home_Charging_Possible == "Yes").astype("float64")})
    inc = raw.Annual_Income_USD.to_numpy(np.int64)
    for suf, k in (("inc", inc), ("inc100", inc // 100)):
        g = d.groupby(k).mean()
        for df in (X, Xte):
            xk = df.Annual_Income_USD.to_numpy(np.int64); xk = xk // 100 if suf == "inc100" else xk
            for c in g.columns: df[f"agg_{suf}_{c}"] = pd.Series(xk).map(g[c]).astype("float32").to_numpy()

def lr_nd(F):
    """LR non-dummy design block: every _te column, NUMS, log1p exact-value freqs, the 4 flags."""
    d = F[[c for c in F.columns if "_te" in c] + NUMS + FLAG4].astype("float32").copy()
    d["l1_fq_inc"] = np.log1p(F.fq_inc.to_numpy(np.float32)); d["l1_fq_km"] = np.log1p(F.fq_km.to_numpy(np.float32))
    return d

def lr_apply(F, spec):
    """LR design matrix on F with the DONOR's mean/std (spec = columns, dummy columns, donor mean, donor std); the column
    lists must match the donor's exactly — get_dummies on a kept categorical dtype always does, anything else asserts."""
    cols, dcols, mu, sd = spec
    nd, dm = lr_nd(F), pd.get_dummies(F[CATS])
    assert list(nd.columns) == cols and list(dm.columns) == dcols
    return np.hstack([((nd - mu) / sd).to_numpy(np.float32), dm.to_numpy(np.float32)])

def fit_basics(F, yf):
    """The two basic models on donor frame F (its _te columns are donor-only by construction). rawlgb on the raw column
    set with no eval_set (a basic model never peeks at a validation fold); lr on the standardised design. The spec pins
    the design columns and F's mean/std so apply frames standardise identically."""
    rc = [c for c in RAWLGB if c in F.columns]
    m1 = lgb.LGBMClassifier(**RAWLGB_PARAMS); m1.fit(F[rc], yf)
    nd, dm = lr_nd(F), pd.get_dummies(F[CATS])
    spec = (list(nd.columns), list(dm.columns), nd.mean(), nd.std().replace(0, 1))
    m2 = LogisticRegression(C=1.0, solver="lbfgs", max_iter=20 if SMOKE else 100, tol=1e-3)
    m2.fit(lr_apply(F, spec), yf)
    return m1, rc, m2, spec

def predict_basics(mods, F):
    """(n, 2) float32 basic logits — rawlgb then lr — from p clipped to [1e-5, 1-1e-5]."""
    m1, rc, m2, spec = mods
    p = np.stack([m1.predict_proba(F[rc])[:, 1], m2.predict_proba(lr_apply(F, spec))[:, 1]], 1)
    return logit(np.clip(p, 1e-5, 1 - 1e-5)).astype(np.float32)

def add_stack_cols(D, S):
    """The five stacked features from the basic logits S (n, 2). The surprisal columns always read D's OWN nested TE, so a
    row's value comes from its own fold's encoding (A's value for A rows, B's for B, C's for C)."""
    D["stk_rawlgb"] = S[:, 0]; D["stk_lr"] = S[:, 1]
    D["stk_surp_exact"] = (logit(np.clip(D.k_inc_exact_teauto.to_numpy(float), 1e-4, 1 - 1e-4)) - S[:, 0]).astype("float32")
    D["stk_surp_100"] = (logit(np.clip(D.k_inc100_teauto.to_numpy(float), 1e-4, 1 - 1e-4)) - S[:, 0]).astype("float32")
    D["stk_lr_minus_raw"] = (S[:, 1] - S[:, 0]).astype("float32")
    assert np.isfinite(D[STACK_COLS].to_numpy()).all()

if __name__ == "__main__":
    t = time.time(); tr, te, o, y, feats = load()
    if SMOKE: tr = tr.iloc[:30000].reset_index(drop=True); y = y[:30000]; te = te.iloc[:5000].reset_index(drop=True)
    X, Xte, K, Kte = build(tr, te, o)
    if DO_AGG:
        agg_frame(X, Xte, tr, te)                    # 14 label-free income aggregates over train+test, once before the folds
        for D in (X, Xte):                           # each row's deviation from its income group (still label-free)
            D["dev_env"] = D.Environmental_Concern_Level - D.agg_inc_env
            D["dev_sub"] = (D.Subsidy_Available.astype(str) == "Yes") - D.agg_inc_sub
            D["dev_rah"] = (D.Range_Anxiety_Level.astype(str) == "High") - D.agg_inc_rah
    refp = f"submissions/oof_v27_hybrid_k{N}_s{SEED}.npy"; ref = roc_auc_score(y, np.load(refp)) if os.path.exists(refp) and not SMOKE else float("nan")
    print(f"{name}: v27 ref OOF {ref:.6f}  features {X.shape[1]}  variant {VARIANT}  stack {'+'.join(STACK_COLS) if DO_STACK else 'none'}", flush=True)
    if DO_STACK: print(f"  rawlgb cols {[c for c in RAWLGB if c in X.columns]}", flush=True)
    cv = StratifiedKFold(N, shuffle=True, random_state=SEED); oof = np.zeros(len(X)); oof_rk = np.zeros(len(X)); pte = np.zeros(len(Xte)); its = []; bas = []
    prm = dict(PARAMS); prm.update(n_estimators=60) if SMOKE else None
    for f, (a, b) in enumerate(cv.split(X, y)):
        A, B, C = fold_frames(X, Xte, K, Kte, y, a, b)
        if DO_STACK:
            S_A = np.full((len(a), 2), np.nan, np.float32); visits = np.zeros(len(a), np.uint8)
            inner = StratifiedKFold(5, shuffle=True, random_state=SEED + 1000 + f)
            for d_loc, r_loc in inner.split(np.zeros(len(a)), y[a]):
                d, r = a[d_loc], a[r_loc]
                Ad, Ar, _ = fold_frames(X, Xte, K, Kte, y, d, r)         # donor-only TE: basics never train on A
                mods = fit_basics(Ad, y[d]); S_A[r_loc] = predict_basics(mods, Ar); visits[r_loc] += 1
                del Ad, Ar, mods; gc.collect()
            assert (visits == 1).all() and np.isfinite(S_A).all()
            mods = fit_basics(A, y[a]); S_B, S_C = predict_basics(mods, B), predict_basics(mods, C)   # refit on all fit rows
            add_stack_cols(A, S_A); add_stack_cols(B, S_B); add_stack_cols(C, S_C)
            bas.append((roc_auc_score(y[b], S_B[:, 0]), roc_auc_score(y[b], S_B[:, 1])))
        ini = [S_A[:, 1], S_B[:, 1], S_C[:, 1]] if INIT_LR else [np.mean([logit(np.clip(D[f"{k}_teauto"].to_numpy(float), 1e-4, 1 - 1e-4)) for k in KEYS], axis=0) for D in (A, B, C)]
        m = lgb.LGBMClassifier(random_state=SEED, **prm)
        m.fit(A, y[a], init_score=ini[0], eval_set=[(B, y[b])], eval_init_score=[ini[1]], eval_metric="auc", callbacks=[lgb.early_stopping(500, verbose=False)])
        pb = expit(m.predict(B, raw_score=True) + ini[1]); pc = expit(m.predict(C, raw_score=True) + ini[2]); its.append(m.best_iteration_)
        oof[b] = pb; oof_rk[b] = rk(pb); pte += rk(pc) / N
        print(f"  fold {f}: auc {roc_auc_score(y[b], pb):.6f}  (init alone {roc_auc_score(y[b], ini[1]):.6f})"
              + (f"  (basic rawlgb {bas[-1][0]:.6f} lr {bas[-1][1]:.6f})" if DO_STACK else "")
              + f"  trees {its[-1]}  A.shape[1] {A.shape[1]}  {time.time()-t:.0f}s", flush=True)
    auc, auc_rk = roc_auc_score(y, oof), roc_auc_score(y, oof_rk)
    print(f"{name} OOF {auc:.6f}  (fold-ranked {auc_rk:.6f})  Δ vs v27 {auc-ref:+.6f}  iters {its}  {time.time()-t:.0f}s", flush=True)
    if SMOKE: sys.exit(0)
    np.save(f"submissions/oof_{name}.npy", oof); pd.DataFrame({"id": te.id, TARGET: pte}).to_csv(f"submissions/{name}.csv", index=False)
    json.dump({"oof_auc": auc, "ref_v27": ref, "iters": its, "n_splits": N, "seed": SEED, "variant": VARIANT,
               "basic_auc": {"rawlgb": float(np.mean([x[0] for x in bas])) if bas else None, "lr": float(np.mean([x[1] for x in bas])) if bas else None},
               "stack_cols": STACK_COLS if DO_STACK else []}, open(f"submissions/{name}.json", "w"), indent=2)
