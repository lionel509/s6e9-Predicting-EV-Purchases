"""Optuna sweep over LightGBM params, on a hard wall-clock deadline.

Searches on 3-fold (fast enough to rank configs), then confirms the winner on the
same 5-fold split the baseline used so the numbers are comparable. Stops starting
new trials at DEADLINE_LOCAL no matter what, so the machine comes back on time.
"""
import datetime as dt
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DEADLINE_LOCAL = dt.datetime.now().replace(hour=13, minute=15, second=0, microsecond=0)
HERE = Path(__file__).parent.parent
DATA, OUT = HERE / "data", HERE / "submissions"
TARGET, ID, SEED = "Will_Buy_EV", "id", 42
BASELINE_OOF = 0.941833

train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
train[TARGET] = train[TARGET].map({"Yes": 1, "No": 0}).astype(int)
features = [c for c in train.columns if c not in (ID, TARGET)]
for c in [c for c in features if not pd.api.types.is_numeric_dtype(train[c])]:
    levels = pd.api.types.union_categoricals(
        [train[c].astype("category"), test[c].astype("category")]).categories
    train[c] = pd.Categorical(train[c], categories=levels)
    test[c] = pd.Categorical(test[c], categories=levels)
X, y, Xt = train[features], train[TARGET], test[features]

FIXED = dict(objective="binary", metric="auc", n_estimators=4000, verbosity=-1, seed=SEED)
search_folds = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(X, y))


def cv(params, folds, collect=False):
    scores, oof, pred, iters = [], np.zeros(len(X)), np.zeros(len(Xt)), []
    for tr, va in folds:
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[tr], y.iloc[tr], eval_X=X.iloc[va], eval_y=y.iloc[va],
              eval_metric="auc", callbacks=[lgb.early_stopping(200, verbose=False)])
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        scores.append(roc_auc_score(y.iloc[va], p))
        iters.append(m.best_iteration_)
        if collect:
            pred += m.predict_proba(Xt)[:, 1] / len(folds)
    return (float(np.mean(scores)), oof, pred, iters) if collect else float(np.mean(scores))


def objective(trial):
    if dt.datetime.now() >= DEADLINE_LOCAL:
        trial.study.stop()
        raise optuna.TrialPruned()
    p = dict(
        FIXED,
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
        num_leaves=trial.suggest_int("num_leaves", 16, 384, log=True),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 20, 500, log=True),
        feature_fraction=trial.suggest_float("feature_fraction", 0.4, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
        bagging_freq=1,
        lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        min_gain_to_split=trial.suggest_float("min_gain_to_split", 0.0, 1.0),
        max_cat_threshold=trial.suggest_int("max_cat_threshold", 8, 64),
    )
    s = cv(p, search_folds)
    print(f"  trial {trial.number:3d}  3f-auc {s:.6f}  lr {p['learning_rate']:.4f} "
          f"leaves {p['num_leaves']} minleaf {p['min_data_in_leaf']}", flush=True)
    return s


print(f"searching until {DEADLINE_LOCAL:%H:%M} local "
      f"({(DEADLINE_LOCAL - dt.datetime.now()).seconds // 60} min)", flush=True)
study = optuna.create_study(
    direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
    study_name="s6e9-lgbm", storage=f"sqlite:///{HERE / 'logs' / 'tuning.db'}", load_if_exists=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)
study.optimize(objective, timeout=max(60, (DEADLINE_LOCAL - dt.datetime.now()).seconds))

done = [t for t in study.trials if t.value is not None]
print(f"\n{len(done)} trials completed. best 3-fold {study.best_value:.6f}")
print("best params:", study.best_params, flush=True)

print("\nconfirming on the baseline's 5-fold split...", flush=True)
best = dict(FIXED, **study.best_params, bagging_freq=1)
full_folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
mean_auc, oof, pred, iters = cv(best, full_folds, collect=True)
oof_auc = roc_auc_score(y, oof)
print(f"tuned OOF AUC {oof_auc:.6f}   baseline {BASELINE_OOF:.6f}   "
      f"delta {oof_auc - BASELINE_OOF:+.6f}")
print(f"best_iters {iters}")
OUT.mkdir(exist_ok=True)
np.save(OUT / "oof_tuned.npy", oof)
pd.DataFrame({ID: test[ID], TARGET: pred}).to_csv(OUT / "tuned.csv", index=False)
print(f"wrote {OUT / 'tuned.csv'}")
print("BEATS BASELINE" if oof_auc > BASELINE_OOF else "does not beat baseline")
print(f"finished {dt.datetime.now():%H:%M:%S}")
