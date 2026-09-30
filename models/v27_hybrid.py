"""v27: megayak's "one LightGBM from raw data" recipe (CV 0.94626 on 10 folds / LB 0.94633), ported onto OUR split so
the OOF is blend-compatible with every oof_*.npy in submissions/. Two feature families on top of Naji's V3:
  - generator leak: low-order digits and moduli of income and commute (x10), a quantisation ladder, exact-value
    frequency over train+test;
  - Naji V3: flags (30k spike, millionaire cliff, dead zone, env hater), original-dataset target means on every
    column, Smooth Keys (income exact / floor 100 / floor 1000, commute floor) + cats + low-card numerics, each
    frequency-encoded and TRIPLE target-encoded with sklearn TargetEncoder (smooth auto / 10 / 100, inner cv=5).
Model: Naji-shaped LightGBM (lr 0.02, depth 5, 32 leaves, colsample 0.3, max_bin 1024), early stopping 500.
OOF is saved as raw probabilities (not fold-ranked) so it compares like-for-like with ours; the fold-ranked
AUC megayak reports is printed alongside.
Usage: python v27_hybrid.py [n_folds=5] [seed=42] [tag]"""
import os, sys, time, json, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from features import load, ORIG
def args():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 5; SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    tag = sys.argv[3] if len(sys.argv) > 3 else ""; return N, SEED, f"v27_hybrid_k{N}_s{SEED}{tag}"
TARGET = "Will_Buy_EV"
CATS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]
NUMS = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned", "Charging_Stations_Near_Home",
        "Charging_Stations_Near_Work", "Environmental_Concern_Level"]
rk = lambda v: rankdata(v) / len(v)

def build(train, test, orig):
    n = len(train)
    df = pd.concat([train[CATS + NUMS], test[CATS + NUMS]], ignore_index=True)
    inc = df.Annual_Income_USD.to_numpy(np.int64)
    km = np.round(df.Daily_Commute_km.to_numpy(float) * 10).astype(np.int64)
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
    K = pd.DataFrame({"k_inc_exact": inc.astype(str), "k_inc100": (inc // 100).astype(str), "k_inc1000": (inc // 1000).astype(str), "k_km_int": (km // 10).astype(str)})
    for c in CATS + ["Age", "Number_of_Cars_Owned", "Charging_Stations_Near_Home", "Charging_Stations_Near_Work", "Environmental_Concern_Level"]:
        K[f"k_{c}"] = df[c].astype(str).to_numpy()
    if os.environ.get("TOKENS") == "1":   # GPT-2 BPE tokens of income: the generator writes numbers token by token (heuljax's LR, issue #1)
        import tiktoken; enc = tiktoken.get_encoding("gpt2"); u, inv = np.unique(inc, return_inverse=True); tk = [enc.encode(" " + str(v)) for v in u]
        K["k_tok1"] = np.array([str(t[0]) for t in tk])[inv]; K["k_tok2"] = np.array(["_".join(map(str, t[:2])) for t in tk])[inv]
        K["k_toklast"] = np.array([f"{len(t)}_{t[-1]}" for t in tk])[inv]
        EXTRA = os.environ.get("EXTRA")   # 2026-09-29 night, issue #1: km = exact commute + its tokens; cross = income tok1 x context
        if EXTRA == "km":
            K["k_km_exact"] = km.astype(str); u, inv = np.unique(df.Daily_Commute_km.to_numpy(float), return_inverse=True)
            tk = [enc.encode(" " + f"{v:g}") for v in u]
            K["k_kmtok1"] = np.array([str(t[0]) for t in tk])[inv]; K["k_kmtoklast"] = np.array([f"{len(t)}_{t[-1]}" for t in tk])[inv]
        elif EXTRA in ("cross", "cross2", "cross3"):
            ctx = ("Subsidy_Available", "Home_Charging_Possible", "City_Type")
            crosses = [("tok1", ctx)] + {"cross": [], "cross3": [("tok2", ctx)],
                                         "cross2": [("tok1", ("Range_Anxiety_Level", "Environmental_Concern_Level", "Current_Car_Type")), ("toklast", ctx)]}[EXTRA]
            for t, cols in crosses:
                for c in cols: K[f"k_x_{t}_{c}"] = K[f"k_{t}"] + "|" + df[c].astype(str).to_numpy()
    for c in K.columns: df[f"{c}_fe"] = K[c].map(K[c].value_counts(normalize=True)).astype("float32").values
    for c in CATS: df[c] = df[c].astype("category")
    cut = lambda x: (x.iloc[:n].reset_index(drop=True), x.iloc[n:].reset_index(drop=True))
    (X, Xte), (Kt, Kte) = cut(df), cut(K)
    return X, Xte, Kt, Kte

PARAMS = dict(n_estimators=20000, learning_rate=0.02, max_depth=5, num_leaves=32, min_child_samples=10, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.3, reg_alpha=0.071, reg_lambda=2.0, max_bin=1024, feature_pre_filter=False, n_jobs=-1, verbose=-1)
def fold_frames(X, Xte, K, Kte, y, a, b, seed=42):
    """Per-fold frames with the triple target encoding added: fit rows inner-CV'd, validation and test from the fit set."""
    A, B, C = X.iloc[a].copy(), X.iloc[b].copy(), Xte.copy()
    for smooth, tg in (("auto", "auto"), (10.0, "10"), (100.0, "100")):
        enc = TargetEncoder(shuffle=True, cv=5, smooth=smooth, random_state=seed)
        ea = enc.fit_transform(K.iloc[a], y[a]); eb, ec = enc.transform(K.iloc[b]), enc.transform(Kte)
        for i, c in enumerate(K.columns):
            A[f"{c}_te{tg}"] = ea[:, i].astype("float32"); B[f"{c}_te{tg}"] = eb[:, i].astype("float32"); C[f"{c}_te{tg}"] = ec[:, i].astype("float32")
    return A, B, C

if __name__ == "__main__":
    N, SEED, name = args(); t = time.time(); tr, te, o, y, feats = load(); X, Xte, K, Kte = build(tr, te, o)
    print(f"{name}: features {X.shape[1]} | TE keys {K.shape[1]} x 3 smoothings  ({time.time()-t:.0f}s)", flush=True)
    cv = StratifiedKFold(N, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X)); oof_rk = np.zeros(len(X)); pte = np.zeros(len(Xte)); iters = []
    for f, (a, b) in enumerate(cv.split(X, y)):
        A, B, C = X.iloc[a].copy(), X.iloc[b].copy(), Xte.copy()
        for smooth, tg in (("auto", "auto"), (10.0, "10"), (100.0, "100")):
            enc = TargetEncoder(shuffle=True, cv=5, smooth=smooth, random_state=42)
            ea = enc.fit_transform(K.iloc[a], y[a]); eb, ec = enc.transform(K.iloc[b]), enc.transform(Kte)
            for i, c in enumerate(K.columns):
                A[f"{c}_te{tg}"] = ea[:, i].astype("float32"); B[f"{c}_te{tg}"] = eb[:, i].astype("float32"); C[f"{c}_te{tg}"] = ec[:, i].astype("float32")
        m = lgb.LGBMClassifier(random_state=SEED, **PARAMS)
        m.fit(A, y[a], eval_set=[(B, y[b])], eval_metric="auc", callbacks=[lgb.early_stopping(500, verbose=False)])
        pb = m.predict_proba(B)[:, 1]; oof[b] = pb; oof_rk[b] = rk(pb); pte += rk(m.predict_proba(C)[:, 1]) / N; iters.append(m.best_iteration_)
        print(f"  fold {f}: auc {roc_auc_score(y[b], pb):.6f}  trees {m.best_iteration_}  {time.time()-t:.0f}s", flush=True)
    auc, auc_rk = roc_auc_score(y, oof), roc_auc_score(y, oof_rk)
    print(f"{name} OOF {auc:.6f}  (fold-ranked {auc_rk:.6f})  iters {iters}  {time.time()-t:.0f}s", flush=True)
    np.save(f"submissions/oof_{name}.npy", oof); pd.DataFrame({"id": te.id, TARGET: pte}).to_csv(f"submissions/{name}.csv", index=False)
    json.dump({"oof_auc": auc, "oof_auc_fold_ranked": auc_rk, "iters": iters, "n_splits": N, "seed": SEED}, open(f"submissions/{name}.json", "w"), indent=2)
