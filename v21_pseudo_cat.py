"""v21: CatBoost as in v8 (income / commute / age as categoricals, its own ordered target statistics, no hand TE)
with the test rows as soft labels (CrossEntropy accepts probabilities). Usage: python v21_pseudo_cat.py <source>"""
import sys, numpy as np, pandas as pd, time
from catboost import CatBoost, Pool
from sklearn.metrics import roc_auc_score
from features import *
src = sys.argv[1]
tr, te, o, y, feats = load(); base = Base(tr, te, o, feats); skf = folds(tr, y)
soft = pd.read_csv(f"submissions/{src}.csv")["Will_Buy_EV"].values; assert 0.1 < soft.mean() < 0.3
ref = roc_auc_score(y, np.load("submissions/oof_v6b_round.npy")); print(f"v6b ref OOF {ref:.6f}   source {src}", flush=True)
def frame(df):
    X = base(df)
    for c in base.cats: X[c] = X[c].astype(str)
    for c in ["Annual_Income_USD", "Daily_Commute_km", "Age"]: X[c + "_cat"] = df[c].astype(str)
    return X
Btr, Bte = frame(tr), frame(te); cat_cols = base.cats + ["Annual_Income_USD_cat", "Daily_Commute_km_cat", "Age_cat"]
P = dict(iterations=4000, learning_rate=0.06, depth=6, l2_leaf_reg=3, loss_function="CrossEntropy", eval_metric="AUC", od_type="Iter", od_wait=150,
         thread_count=12, random_seed=42, verbose=0, allow_writing_files=False)
oof = np.zeros(len(tr)); pred = np.zeros(len(te)); its = []; t = time.time()
for a, b in skf:
    Xall = pd.concat([Btr.iloc[a], Bte], ignore_index=True); yall = np.r_[y[a], soft]
    mdl = CatBoost(P); mdl.fit(Pool(Xall, yall, cat_features=cat_cols), eval_set=Pool(Btr.iloc[b], y[b], cat_features=cat_cols), use_best_model=True)
    oof[b] = mdl.predict(Pool(Btr.iloc[b], cat_features=cat_cols), prediction_type="Probability")[:, 1] if mdl.predict(Pool(Btr.iloc[b][:2], cat_features=cat_cols), prediction_type="Probability").ndim == 2 else mdl.predict(Pool(Btr.iloc[b], cat_features=cat_cols), prediction_type="Probability")
    pt = mdl.predict(Pool(Bte, cat_features=cat_cols), prediction_type="Probability"); pred += (pt[:, 1] if pt.ndim == 2 else pt) / 5; its.append(mdl.get_best_iteration())
    print(f"  fold auc {roc_auc_score(y[b], oof[b]):.6f}  {time.time()-t:.0f}s", flush=True)
auc = roc_auc_score(y, oof); name = "v21_pseudo_cat"
print(f"{name} OOF {auc:.6f}  Δ vs v6b {auc-ref:+.6f}  iters {its}  {time.time()-t:.0f}s", flush=True)
np.save(f"submissions/oof_{name}.npy", oof); pd.DataFrame({"id": te.id, "Will_Buy_EV": pred}).to_csv(f"submissions/{name}.csv", index=False)
