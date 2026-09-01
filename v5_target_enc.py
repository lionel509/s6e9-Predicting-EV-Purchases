"""v5: nested out-of-fold target encoding of the high-cardinality numerics, on top of
v3 count-encoding and v4 original-row lookup. Income has 13k distinct values and is a
generator seed; an 18-leaf tree cannot carve 13k levels, a per-value target rate can."""
import pandas as pd, numpy as np, lightgbm as lgb, time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
tr = pd.read_csv("data/train.csv"); te = pd.read_csv("data/test.csv")
o = pd.read_csv("data/orig/EV_Adoption_and_Range_Anxiety_Dataset.csv").drop(columns="Buyer_ID")
y = (tr["Will_Buy_EV"]=="Yes").astype(int).values
feats = [c for c in tr.columns if c not in ("id","Will_Buy_EV")]
allX = pd.concat([tr[feats], te[feats], o[feats]], ignore_index=True)
cats = [c for c in feats if str(allX[c].dtype) in ("object","str","string")]
synth = pd.concat([tr[feats], te[feats]], ignore_index=True)
TE_COLS = ["Annual_Income_USD","Daily_Commute_km","Age"]; M = 20
def prep(df):
    X = df[feats].copy()
    for c in cats: X[c] = pd.Categorical(X[c], categories=sorted(allX[c].dropna().unique()))
    for c in ["Annual_Income_USD","Daily_Commute_km"]: X[c+"_cnt"] = df[c].map(synth[c].value_counts()).astype(float)
    for c,nm in [("Annual_Income_USD","inc"),("Daily_Commute_km","com")]:
        g = o.groupby(c)["Will_Buy_EV"].agg(cnt="size", rate=lambda s:(s=="Yes").mean())
        X[nm+"_orig_cnt"] = df[c].map(g["cnt"]).fillna(0).astype(float); X[nm+"_orig_rate"] = df[c].map(g["rate"]).astype(float)
    return X
def te_fit(keys, target, prior):
    g = pd.DataFrame({"k":keys.values,"y":target}).groupby("k")["y"].agg(["sum","size"])
    return (g["sum"] + prior*M)/(g["size"]+M)
def add_te(Xfit, yfit, Xapply_list, inner=5, seed=0):
    """TE for the fit rows via inner OOF, for apply rows from the whole fit set."""
    prior = yfit.mean(); Xfit = Xfit.copy(); outs=[x.copy() for x in Xapply_list]
    for c in TE_COLS:
        col = np.zeros(len(Xfit))
        for a,b in StratifiedKFold(inner, shuffle=True, random_state=seed).split(Xfit, yfit):
            col[b] = Xfit[c].iloc[b].map(te_fit(Xfit[c].iloc[a], yfit[a], prior)).fillna(prior).values
        Xfit[c+"_te"] = col
        full = te_fit(Xfit[c], yfit, prior)
        for x in outs: x[c+"_te"] = x[c].map(full).fillna(prior).values
    return Xfit, outs
params = dict(learning_rate=0.05, num_leaves=18, min_data_in_leaf=152, feature_fraction=0.463, bagging_fraction=0.957,
              bagging_freq=1, lambda_l1=0.0034, lambda_l2=0.0022, min_gain_to_split=0.329, max_cat_threshold=37, verbose=-1, n_estimators=4000)
if __name__ == "__main__":
    Xtr, Xte = prep(tr), prep(te)
    skf = list(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y))
    oof = np.zeros(len(tr)); pred = np.zeros(len(te)); its=[]; t=time.time()
    for f,(a,b) in enumerate(skf):
        Xa, (Xb, Xt) = add_te(Xtr.iloc[a], y[a], [Xtr.iloc[b], Xte])
        m = lgb.LGBMClassifier(**params); m.fit(Xa, y[a], eval_set=[(Xb, y[b])], callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[b] = m.predict_proba(Xb)[:,1]; pred += m.predict_proba(Xt)[:,1]/len(skf); its.append(m.best_iteration_)
        print(f"fold {f}: {roc_auc_score(y[b], oof[b]):.6f}  iters {m.best_iteration_}", flush=True)
    auc = roc_auc_score(y, oof); print(f"v5 OOF AUC {auc:.6f}  iters {its}  {time.time()-t:.0f}s")
    print("importances:", pd.Series(m.feature_importances_, index=Xa.columns).sort_values(ascending=False).head(10).to_dict())
    np.save("submissions/oof_v5_te.npy", oof)
    pd.DataFrame({"id": te.id, "Will_Buy_EV": pred}).to_csv("submissions/v5_te.csv", index=False)
    ref = np.load("submissions/oof_tuned.npy") if __import__("os").path.exists("submissions/oof_tuned.npy") else None
    if ref is not None: print(f"Δ vs tuned (run 3, 0.942287): {auc-0.942287:+.6f}")
