"""v4: the original dataset (itzzomkar/ev-adoption-behavior-and-range-anxiety, 10k rows) three ways: concatenated into training, as a per-value label lookup, and both. Paired against v3."""
import pandas as pd, numpy as np, lightgbm as lgb, time, glob
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
S="."
tr = pd.read_csv("data/train.csv"); te = pd.read_csv("data/test.csv")
o = pd.read_csv("data/orig/EV_Adoption_and_Range_Anxiety_Dataset.csv").drop(columns="Buyer_ID")
print("orig NaNs:", o.isna().sum()[o.isna().sum()>0].to_dict())
y = (tr["Will_Buy_EV"]=="Yes").astype(int).values; yo = (o["Will_Buy_EV"]=="Yes").astype(int).values
feats = [c for c in tr.columns if c not in ("id","Will_Buy_EV")]
allX = pd.concat([tr[feats], te[feats], o[feats]], ignore_index=True)
cats = [c for c in feats if str(allX[c].dtype) in ("object","str","string")]
def prep(df):
    X = df[feats].copy()
    for c in cats: X[c] = pd.Categorical(X[c], categories=sorted(allX[c].dropna().unique()))
    return X
synth = pd.concat([tr[feats], te[feats]], ignore_index=True)
def add_counts(X, src):
    for c in ["Annual_Income_USD","Daily_Commute_km"]:
        X[c+"_cnt"] = src[c].map(synth[c].value_counts()).astype(float)
    return X
def add_lookup(X, src):
    for c,nm in [("Annual_Income_USD","inc"),("Daily_Commute_km","com")]:
        g = o.groupby(c)["Will_Buy_EV"].agg(cnt="size", rate=lambda s:(s=="Yes").mean())
        X[nm+"_orig_cnt"] = src[c].map(g["cnt"]).fillna(0).astype(float)
        X[nm+"_orig_rate"] = src[c].map(g["rate"]).astype(float)
    return X
Xtr_c = add_counts(prep(tr), tr); Xo_c = add_counts(prep(o), o)
Xtr_l = add_lookup(add_counts(prep(tr), tr), tr); Xo_l = add_lookup(add_counts(prep(o), o), o)
params = dict(learning_rate=0.05, num_leaves=18, min_data_in_leaf=152, feature_fraction=0.463, bagging_fraction=0.957,
              bagging_freq=1, lambda_l1=0.0034, lambda_l2=0.0022, min_gain_to_split=0.329, max_cat_threshold=37, verbose=-1, n_estimators=3000)
skf = list(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y))
ref = np.load("submissions/oof_v3_count_enc.npy")[:,1]; print(f"ref +count_enc      OOF {roc_auc_score(y,ref):.6f}")
res={}
for name, Xtr, Xo, concat in [("+count +concat_orig", Xtr_c, Xo_c, True), ("+count +orig_lookup", Xtr_l, Xo_l, False), ("+count +lookup +concat", Xtr_l, Xo_l, True)]:
    oof=np.zeros(len(tr)); t=time.time(); its=[]
    for a,b in skf:
        Xa, ya = (pd.concat([Xtr.iloc[a], Xo]), np.r_[y[a], yo]) if concat else (Xtr.iloc[a], y[a])
        m = lgb.LGBMClassifier(**params); m.fit(Xa, ya, eval_set=[(Xtr.iloc[b], y[b])], callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[b]=m.predict_proba(Xtr.iloc[b])[:,1]; its.append(m.best_iteration_)
    res[name]=oof; print(f"{name:24s} OOF {roc_auc_score(y,oof):.6f}  Δ vs +count {roc_auc_score(y,oof)-roc_auc_score(y,ref):+.6f}  iters {its}  {time.time()-t:.0f}s", flush=True)
np.save("submissions/oof_v4_variants.npy", np.c_[[res[k] for k in res]].T)
if "+count +orig_lookup" in res:
    imp = pd.Series(m.feature_importances_, index=Xtr_l.columns).sort_values(ascending=False); print("last model importances:", imp.head(8).to_dict())
