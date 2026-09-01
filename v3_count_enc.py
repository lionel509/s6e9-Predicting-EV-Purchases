"""v3: value-frequency (count) encoding of income and commute, computed over train+test. Paired 5-fold A/B against the tuned params."""
import pandas as pd, numpy as np, lightgbm as lgb, time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
tr = pd.read_csv("data/train.csv"); te = pd.read_csv("data/test.csv")
y = (tr["Will_Buy_EV"]=="Yes").astype(int).values
feats = [c for c in tr.columns if c not in ("id","Will_Buy_EV")]
full = pd.concat([tr[feats], te[feats]], ignore_index=True)
# --- variants
def base(df):
    X = df.copy()
    for c in X.columns:
        if str(X[c].dtype) in ("object","str","string"): X[c] = X[c].astype("category")
    return X
def with_counts(df):
    X = base(df)
    for c in ["Annual_Income_USD","Daily_Commute_km"]:
        X[c+"_cnt"] = df[c].map(full[c].value_counts()).astype(float)
    return X
def ordinal(df):
    X = with_counts(df)
    X["Range_Anxiety_Level"] = df["Range_Anxiety_Level"].map({"Low":0,"Medium":1,"High":2}).astype(float)
    return X
params = dict(learning_rate=0.05, num_leaves=18, min_data_in_leaf=152, feature_fraction=0.463, bagging_fraction=0.957,
              bagging_freq=1, lambda_l1=0.0034, lambda_l2=0.0022, min_gain_to_split=0.329, max_cat_threshold=37, verbose=-1, n_estimators=3000)
skf = list(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y))
res = {}
for name, fn in [("tuned_raw", base), ("+count_enc", with_counts), ("+count+ordinal", ordinal)]:
    X = fn(full).iloc[:len(tr)]
    oof = np.zeros(len(tr)); t=time.time(); its=[]
    for a,b in skf:
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[a], y[a], eval_set=[(X.iloc[b], y[b])], callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[b] = m.predict_proba(X.iloc[b])[:,1]; its.append(m.best_iteration_)
    res[name]=oof
    print(f"{name:16s} OOF AUC {roc_auc_score(y,oof):.6f}  iters {its}  {time.time()-t:.0f}s", flush=True)
b = roc_auc_score(y,res["tuned_raw"])
for k,v in res.items(): print(f"  Δ {k:16s} {roc_auc_score(y,v)-b:+.6f}")
np.save("submissions/oof_v3_count_enc.npy", np.c_[res["tuned_raw"],res["+count_enc"],res["+count+ordinal"]])
