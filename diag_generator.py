"""Diagnostics on the generator: id order, income-join column agreement with the original, 11-feature combo key, and public-leaderboard noise from OOF slices."""
import pandas as pd, numpy as np, glob
from sklearn.metrics import roc_auc_score
S="."
tr = pd.read_csv("data/train.csv"); te = pd.read_csv("data/test.csv")
o = pd.read_csv("data/orig/EV_Adoption_and_Range_Anxiety_Dataset.csv").drop(columns="Buyer_ID")
y = (tr["Will_Buy_EV"]=="Yes").astype(int).values
feats = [c for c in tr.columns if c not in ("id","Will_Buy_EV")]

# (c) id order
print("== id order ==")
print(f"AUC(target ~ id) = {roc_auc_score(y, tr.id):.4f}; ids contiguous train 0..{tr.id.max()}, test {te.id.min()}..{te.id.max()}")
blk = pd.Series(y).groupby(tr.id//50000).mean(); print("target rate by 50k id block:", blk.round(4).tolist())

# (h) does the income-join recover the original row's other columns?
print("\n== income join to original: column agreement ==")
ou = o.drop_duplicates("Annual_Income_USD", keep=False).set_index("Annual_Income_USD")   # incomes unique in orig
j = tr.join(ou, on="Annual_Income_USD", rsuffix="_o", how="inner"); print(f"train rows joined to a unique-income orig row: {len(j):,}")
for c in feats+["Will_Buy_EV"]:
    if c=="Annual_Income_USD": continue
    agree = (j[c]==j[c+"_o"]).mean()
    # chance agreement if independent: sum p_i^2 over marginal
    p = tr[c].value_counts(normalize=True); chance = (p**2).sum()
    print(f"  {c:30s} agree {agree:.3f}   chance {chance:.3f}   lift {agree-chance:+.3f}")
# same for commute join
print("\n== commute join: label agreement ==")
oc = o.groupby("Daily_Commute_km")["Will_Buy_EV"].apply(lambda s:(s=="Yes").mean())
print(f"  AUC(target ~ orig rate by commute) = {roc_auc_score(y, tr.Daily_Commute_km.map(oc).fillna(0.175)):.4f}")
oi = o.groupby("Annual_Income_USD")["Will_Buy_EV"].apply(lambda s:(s=="Yes").mean())
print(f"  AUC(target ~ orig rate by income)  = {roc_auc_score(y, tr.Annual_Income_USD.map(oi).fillna(0.175)):.4f}")

# (e) high-order combo key: all low-card features (+Age), 2-fold holdout target rate
print("\n== 11-feature combo key (everything but income & commute) ==")
lowc = [c for c in feats if c not in ("Annual_Income_USD","Daily_Commute_km")]
k = tr[lowc].astype(str).agg("|".join, axis=1); kt = te[lowc].astype(str).agg("|".join, axis=1)
print(f"  unique keys train {k.nunique():,}; test rows with key seen in train {kt.isin(set(k)).mean():.1%}")
rng = np.random.RandomState(0); h = rng.rand(len(tr))<0.5
rate = pd.Series(y[h]).groupby(k[h].values).agg(["mean","size"])
m = k[~h].map(rate["mean"]); n = k[~h].map(rate["size"]).fillna(0)
prior=y.mean(); sm = (m.fillna(prior)*n + prior*20)/(n+20)
print(f"  holdout AUC of smoothed combo-key rate alone: {roc_auc_score(y[~h], sm):.4f}  (coverage {m.notna().mean():.1%})")

# (9) public LB noise: SE of AUC on a 20% test slice (~57k rows), bootstrap from best OOF
print("\n== leaderboard noise ==")
oof = np.load("submissions/oof_v4_variants.npy")[:,1]  # +count +orig_lookup
n_pub = int(0.2*len(te)); aucs=[roc_auc_score(y[i], oof[i]) for i in (rng.choice(len(tr), n_pub, replace=False) for _ in range(200))]
print(f"  full OOF AUC {roc_auc_score(y,oof):.5f}; on random {n_pub:,}-row slices: sd = {np.std(aucs):.5f}, 95% band ±{1.96*np.std(aucs):.5f}")
