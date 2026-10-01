"""v10: the generator's recipe as features, on top of v6b. Chris Deotte reverse-engineered the original
dataset's generator (NumPy RandomState seed 101, 100% column match): Range_Anxiety is a thresholded worry
score and Will_Buy_EV a thresholded buy score, both linear in the row's own columns, plus a normal wobble.
Leak-free (each row uses only its own columns); trees otherwise approximate the two lines with many splits.
Paired vs v6b on the same folds. Deotte's XGB starter reports the gain at a few ten-thousandths."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from features import *

def recipe(df):
    home = (df.Home_Charging_Possible == "Yes").astype(int); sub = (df.Subsidy_Available == "Yes").astype(int)
    R = pd.DataFrame(index=df.index)
    R["worry_score"] = df.Daily_Commute_km - 5 * df.Charging_Stations_Near_Home - 5 * df.Charging_Stations_Near_Work - 150 * home
    R["buy_score"] = (1.2 * df.Annual_Income_USD / 1e5 + 0.6 * df.Environmental_Concern_Level + 2.0 * sub
                      - 1.0 * (df.Range_Anxiety_Level == "Medium").astype(int) - 3.0 * (df.Range_Anxiety_Level == "High").astype(int))
    R["chargers_total"] = df.Charging_Stations_Near_Home + df.Charging_Stations_Near_Work
    R["income_x_subsidy"] = df.Annual_Income_USD / 1e5 * sub
    R["concern_x_subsidy"] = df.Environmental_Concern_Level * sub
    return R

if __name__ == "__main__":
    tr, te, o, y, feats = load(); base = Base(tr, te, o, feats); skf = folds(tr, y)
    Btr, Bte = base(tr).join(recipe(tr)), base(te).join(recipe(te))
    ref = roc_auc_score(y, np.load("submissions/oof_v6b_round.npy")); print(f"v6b ref OOF {ref:.6f}")
    print(f"recipe alone (buy_score, no training) AUC {roc_auc_score(y, Btr.buy_score):.4f}   # Deotte reports ~0.938", flush=True)
    lgbm = lambda: lgb.LGBMClassifier(**LGB_PARAMS)
    es = lambda Xb, yb: dict(eval_set=[(Xb, yb)], callbacks=[lgb.early_stopping(150, verbose=False)])
    _, _, auc = run_cv(lgbm, tr, te, y, Btr, Bte, V5_SPECS + ROUND_SPECS, skf, "v10_recipe", es); print(f"   Δ vs v6b {auc-ref:+.6f}", flush=True)
