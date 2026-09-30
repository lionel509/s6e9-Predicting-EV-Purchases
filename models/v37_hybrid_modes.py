"""v37: two new modes on the v27 hybrid recipe, with v32's modes copied along so they stay runnable from here (outputs
are now named v37_*):
  tecv     inner TargetEncoder cv 5 -> INNER (default 10): fit rows get encodings closer to what validation/test get
  wobble   sample weight ALPHA (default 2) on rows inside the generator's undecided band 4.5 < buy score < 6.5
  rank     XGBoost rank:pairwise (RankNet loss, PAIRS random pairs per row over the whole fit set) — a direct AUC surrogate
  linear   LightGBM linear_tree=True: a linear model in every leaf (the recipe is linear in income inside each band)
  extra    LightGBM extra_trees=True: random split thresholds — a cheap decorrelation lever on the same frame
  additive interaction_constraints=[[i] for i in range(ncols)]: every tree uses exactly one feature, an additive booster
           matching the generator's additive label rule (a public notebook measured +0.0012 from this constraint alone);
           n_estimators raised to 60000 (ES 500 stays).
  mono     additive's structure with the sign of the generator's effects instead of its additivity: monotone_constraints
           (method 'advanced') — +1 on every _te column and _org_mean column (more leaked signal, more buy), +1 on
           Environmental_Concern_Level and is_millionaire_cliff, -1 on is_30k_spike / is_dead_zone / is_env_hater, 0 on
           everything else (categoricals, raw income and the inc_q* ladder must stay unconstrained). n_estimators stays
           the normal 20000 — the trees still interact, they just cannot get a sign wrong.
  agg      label-free income aggregates over the raw train+test rows: groupby income exact and income//100 of mean
           Environmental_Concern_Level, share Subsidy_Available Yes, share Range_Anxiety_Level High / Low, mean
           Daily_Commute_km, mean Age, share Home_Charging_Possible Yes — mapped on as agg_inc_* / agg_inc100_* float32
           columns. No label touches them so nothing needs nesting inside the folds.
additive / mono / agg all take the optional 4th arg = an init key (v34's idea) so boosting can start from that key's
nested encoding instead of the base rate — name suffix _init<key without k_> when given.
Everything else — frame, split, TE seed, Naji-shaped params, ES 500 — is v27's, so the OOF compares like-for-like.
Usage: python v37_hybrid_modes.py MODE [n_folds=5] [seed=42] [param]
       param = INNER (tecv) | ALPHA (wobble) | PAIRS (rank) | LINEAR_LAMBDA (linear) | init key (additive, mono, agg).
       SMOKE=1 env: 30k rows, 60 trees."""
import os, sys, time, json, numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from scipy.special import logit, expit
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from features import load
from v27_hybrid import build, PARAMS, TARGET
rk = lambda v: rankdata(v) / len(v)
MODE = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 5; SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 42
P = sys.argv[4] if len(sys.argv) > 4 else None
INNER = int(P) if MODE == "tecv" and P else 10
ALPHA = float(P) if MODE == "wobble" and P else 2.0
PAIRS = int(P) if MODE == "rank" and P else 8
LLAM = float(P) if MODE == "linear" and P else 0.0
INIT = P if MODE in ("additive", "mono", "agg") and P else None
isfx = f"_init{INIT.replace('k_', '')}" if INIT else ""
suffix = {"tecv": f"_cv{INNER}", "wobble": f"_a{ALPHA:g}", "rank": f"_p{PAIRS}", "linear": f"_l{LLAM:g}", "extra": "",
          "additive": isfx, "mono": isfx, "agg": isfx}[MODE]
SMOKE = os.environ.get("SMOKE") == "1"
name = f"v37_{MODE}{suffix}{'_tok' if os.environ.get('TOKENS') == '1' else ''}_k{N}_s{SEED}" + ("_smoke" if SMOKE else "")

def frames(X, Xte, K, Kte, y, a, b, inner=5):
    """v27's fold_frames with the inner cv exposed. TE seed stays 42 like v27's main loop."""
    A, B, C = X.iloc[a].copy(), X.iloc[b].copy(), Xte.copy()
    for smooth, tg in (("auto", "auto"), (10.0, "10"), (100.0, "100")):
        enc = TargetEncoder(shuffle=True, cv=inner, smooth=smooth, random_state=42)
        ea = enc.fit_transform(K.iloc[a], y[a]); eb, ec = enc.transform(K.iloc[b]), enc.transform(Kte)
        for i, c in enumerate(K.columns):
            A[f"{c}_te{tg}"] = ea[:, i].astype("float32"); B[f"{c}_te{tg}"] = eb[:, i].astype("float32"); C[f"{c}_te{tg}"] = ec[:, i].astype("float32")
    return A, B, C

def buy_score(df):
    """Deotte's reverse-engineered generator: buy = 1.2*income/1e5 + 0.6*concern + 2*subsidy - 1*Medium - 3*High + N(0,1) > 5.5."""
    return (1.2 * df.Annual_Income_USD.to_numpy(float) / 1e5 + 0.6 * df.Environmental_Concern_Level.to_numpy(float)
            + 2.0 * (df.Subsidy_Available.to_numpy() == "Yes") - 1.0 * (df.Range_Anxiety_Level.to_numpy() == "Medium") - 3.0 * (df.Range_Anxiety_Level.to_numpy() == "High"))

MONO_UP = ["Environmental_Concern_Level", "is_millionaire_cliff"]; MONO_DOWN = ["is_30k_spike", "is_dead_zone", "is_env_hater"]
def mono_vec(cols):
    """Generator sign pattern as monotone_constraints: +1 on every target encoding and original mean (the leaked signal
    and the millionaire cliff raise buy), -1 on the three spurious flags, 0 on everything else — categoricals, raw income
    and the inc_q* ladders must stay unconstrained. Prints the +1/-1/0 counts and which named flags were found."""
    s = set(cols); up = [c for c in MONO_UP if c in s]; down = [c for c in MONO_DOWN if c in s]
    vec = [1 if ("_te" in c or c.endswith("_org_mean") or c in MONO_UP) else -1 if c in MONO_DOWN else 0 for c in cols]
    print(f"  mono constraints: +1 {vec.count(1)}  -1 {vec.count(-1)}  0 {vec.count(0)}  named flags up {up} down {down}  missing {[c for c in MONO_UP + MONO_DOWN if c not in s]}", flush=True)
    return vec

def agg_frame(X, Xte, tr, te):
    """Label-free income aggregates over the raw train+test rows, mapped on as agg_inc_* / agg_inc100_* float32 columns.
    env/sub/rah/ral/km/age/home = mean concern, share subsidy Yes, share range High, share range Low, mean commute, mean
    age, share home charging Yes. No labels used (the frame is built before any y enters), so no fold nesting needed."""
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

if __name__ == "__main__":
    t = time.time(); tr, te, o, y, feats = load()
    if SMOKE: tr = tr.iloc[:30000].reset_index(drop=True); y = y[:30000]; te = te.iloc[:5000].reset_index(drop=True)
    X, Xte, K, Kte = build(tr, te, o)
    if MODE == "agg": agg_frame(X, Xte, tr, te)
    refp = f"submissions/oof_v27_hybrid_k{N}_s{SEED}.npy"; ref = roc_auc_score(y, np.load(refp)) if os.path.exists(refp) and not SMOKE else float("nan")
    w = np.ones(len(X))
    if MODE == "wobble":
        s = buy_score(tr); band = (s > 4.5) & (s < 6.5); w[band] = ALPHA
        print(f"  wobble band: {band.mean():.3f} of rows, buy rate inside {y[band].mean():.3f} outside {y[~band].mean():.3f}, weight {ALPHA}", flush=True)
    print(f"{name}: v27 ref OOF {ref:.6f}  features {X.shape[1]}  ({time.time()-t:.0f}s)", flush=True)
    cv = StratifiedKFold(N, shuffle=True, random_state=SEED); oof = np.zeros(len(X)); oof_rk = np.zeros(len(X)); pte = np.zeros(len(Xte)); its = []
    for f, (a, b) in enumerate(cv.split(X, y)):
        A, B, C = frames(X, Xte, K, Kte, y, a, b, inner=INNER if MODE == "tecv" else 5)
        if MODE == "rank":
            # random query groups of GSIZE rows: XGBoost parallelises the pairwise gradients over queries, so one 535k-row
            # query runs on a single core (killed after 12 min without a fold); random groups keep the pairs random overall
            GSIZE = 1000; rng = np.random.RandomState(SEED); pa, pbm = rng.permutation(len(a)), rng.permutation(len(b))
            da = xgb.DMatrix(A.iloc[pa], label=y[a][pa], qid=(np.arange(len(a)) // GSIZE).astype(np.uint32), enable_categorical=True)
            db = xgb.DMatrix(B.iloc[pbm], label=y[b][pbm], qid=(np.arange(len(b)) // GSIZE).astype(np.uint32), enable_categorical=True)
            dc = xgb.DMatrix(C, enable_categorical=True)
            prm = dict(objective="rank:pairwise", lambdarank_pair_method="mean", lambdarank_num_pair_per_sample=PAIRS, tree_method="hist",
                       max_depth=5, eta=0.05, subsample=0.9, colsample_bytree=0.6, min_child_weight=50, reg_lambda=2.0, max_cat_to_onehot=1,
                       eval_metric="auc", nthread=12, seed=SEED)
            bst = xgb.train(prm, da, 60 if SMOKE else 6000, evals=[(db, "val")], early_stopping_rounds=200, verbose_eval=False)
            ir = (0, bst.best_iteration + 1); pb = np.empty(len(b)); pb[pbm] = bst.predict(db, iteration_range=ir); pc = bst.predict(dc, iteration_range=ir); its.append(bst.best_iteration)
        else:
            prm = dict(PARAMS)
            if MODE == "linear": prm.update(linear_tree=True, linear_lambda=LLAM)
            if MODE == "extra": prm.update(extra_trees=True)
            if MODE == "additive": prm.update(n_estimators=60000, interaction_constraints=[[i] for i in range(A.shape[1])])
            if MODE == "mono":
                v = mono_vec(A.columns); assert len(v) == A.shape[1]
                prm.update(monotone_constraints=v, monotone_constraints_method="advanced")
            if SMOKE: prm.update(n_estimators=60)
            m = lgb.LGBMClassifier(random_state=SEED, **prm)
            if MODE in ("additive", "mono", "agg") and INIT:
                ini = [logit(np.clip(D[f"{INIT}_teauto"].to_numpy(float), 1e-4, 1 - 1e-4)) for D in (A, B, C)]
                m.fit(A, y[a], init_score=ini[0], eval_set=[(B, y[b])], eval_init_score=[ini[1]], eval_metric="auc", callbacks=[lgb.early_stopping(500, verbose=False)])
                pb = expit(m.predict(B, raw_score=True) + ini[1]); pc = expit(m.predict(C, raw_score=True) + ini[2])
            else:
                m.fit(A, y[a], sample_weight=w[a] if MODE == "wobble" else None, eval_set=[(B, y[b])], eval_metric="auc", callbacks=[lgb.early_stopping(500, verbose=False)])
                pb = m.predict_proba(B)[:, 1]; pc = m.predict_proba(C)[:, 1]
            its.append(m.best_iteration_)
        oof[b] = pb; oof_rk[b] = rk(pb); pte += rk(pc) / N
        print(f"  fold {f}: auc {roc_auc_score(y[b], pb):.6f}  trees {its[-1]}  {time.time()-t:.0f}s", flush=True)
    if MODE == "rank": oof = oof_rk   # pairwise scores have no shared scale across folds: save the fold-ranked OOF (compare with v27's fold-ranked AUC)
    auc, auc_rk = roc_auc_score(y, oof), roc_auc_score(y, oof_rk); print(f"{name} OOF {auc:.6f}  (fold-ranked {auc_rk:.6f})  Δ vs v27 {auc-ref:+.6f}  iters {its}  {time.time()-t:.0f}s", flush=True)
    if SMOKE: sys.exit(0)
    np.save(f"submissions/oof_{name}.npy", oof); pd.DataFrame({"id": te.id, TARGET: pte}).to_csv(f"submissions/{name}.csv", index=False)
    json.dump({"oof_auc": auc, "ref_v27": ref, "iters": its, "n_splits": N, "seed": SEED, "mode": MODE, "param": P}, open(f"submissions/{name}.json", "w"), indent=2)
