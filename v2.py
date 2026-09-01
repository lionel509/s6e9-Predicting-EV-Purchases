"""S6E9 v2: baseline + the two floor-clip flags + targeted interactions.

Deliberately dumb. Its only job is to put a number on the board so every later
change has a reference point. Schema-agnostic — it reads the columns off the
frame rather than hardcoding them.
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "submissions"
TARGET = "Will_Buy_EV"
ID = "id"
SEED = 42
FOLDS = 5

if not (DATA / "train.csv").exists():
    sys.exit(f"no data in {DATA} — run: uv run kaggle competitions download -c playground-series-s6e9 -p data")

train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
print(f"train {train.shape}  test {test.shape}")
train[TARGET] = train[TARGET].map({"Yes": 1, "No": 0}).astype(int)  # target ships as Yes/No strings
print(f"positive rate {train[TARGET].mean():.4f}")

def add_features(df):
    df = df.copy()
    # the only real generator artifacts: two clipped floors (see the profiling run)
    df["income_at_floor"] = (df.Annual_Income_USD == 30000).astype(int)
    df["commute_at_floor"] = (df.Daily_Commute_km == 5.0).astype(int)
    # low-cardinality frame: interactions, not digits
    df["stations_total"] = df.Charging_Stations_Near_Home + df.Charging_Stations_Near_Work
    df["stations_diff"] = df.Charging_Stations_Near_Home - df.Charging_Stations_Near_Work
    df["income_per_car"] = df.Annual_Income_USD / df.Number_of_Cars_Owned
    df["income_per_km"] = df.Annual_Income_USD / df.Daily_Commute_km
    df["commute_per_car"] = df.Daily_Commute_km / df.Number_of_Cars_Owned
    df["concern_x_stations"] = df.Environmental_Concern_Level * df.stations_total
    df["concern_x_income"] = df.Environmental_Concern_Level * df.Annual_Income_USD
    df["km_x_stations"] = df.Daily_Commute_km * df.stations_total
    return df

train, test = add_features(train), add_features(test)
features = [c for c in train.columns if c not in (ID, TARGET)]
cats = [c for c in features if not pd.api.types.is_numeric_dtype(train[c])]  # pandas 3 says "str", not "object"
for c in cats:  # shared categories across train/test so codes line up
    levels = pd.api.types.union_categoricals(
        [train[c].astype("category"), test[c].astype("category")]
    ).categories
    train[c] = pd.Categorical(train[c], categories=levels)
    test[c] = pd.Categorical(test[c], categories=levels)
print(f"{len(features)} features, {len(cats)} categorical: {cats}")

X, y, Xt = train[features], train[TARGET], test[features]
oof = np.zeros(len(X))
pred = np.zeros(len(Xt))
params = dict(
    objective="binary", metric="auc", learning_rate=0.04, num_leaves=64,
    feature_fraction=0.7, bagging_fraction=0.75, bagging_freq=1,
    lambda_l1=0.2, lambda_l2=0.4, min_data_in_leaf=100,
    n_estimators=3000, verbosity=-1, seed=SEED,
)

for fold, (tr, va) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(X, y)):
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X.iloc[tr], y.iloc[tr],
        eval_X=X.iloc[va], eval_y=y.iloc[va], eval_metric="auc",
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    oof[va] = model.predict_proba(X.iloc[va])[:, 1]
    pred += model.predict_proba(Xt)[:, 1] / FOLDS
    print(f"fold {fold}  auc {roc_auc_score(y.iloc[va], oof[va]):.6f}  best_iter {model.best_iteration_}")

print(f"\nOOF AUC {roc_auc_score(y, oof):.6f}   <-- this is a LOCAL number, not the leaderboard")
OUT.mkdir(exist_ok=True)
np.save(OUT / "oof_v2.npy", oof)
sub = pd.DataFrame({ID: test[ID], TARGET: pred})
sub.to_csv(OUT / "v2.csv", index=False)
print(f"wrote {OUT / 'v2.csv'}  ({len(sub)} rows, expected 286571)")
