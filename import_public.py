"""Copy the public OOF / test pairs in data/public/ into the submissions/ convention (oof_<name>.npy + <name>.csv)
so assemble.py can use them as groups. Names are prefixed pub_ so their provenance is visible in every blend line.
Sources (all on StratifiedKFold(5, shuffle, 42) or a strict superset partition — see the hub note):
  pub_megayak10  megayak, one LightGBM from raw data, 10-fold, OOF 0.946264 (fold-ranked, ES on the held-out fold)
  pub_naji_v3    Naji, Pure LGBM V3, OOF 0.946064
  pub_sergey     Sergey, focal-loss LGBM, OOF 0.945329 (logits)
  pub_naji_01blend  Naji's six-notebook rank blend (Dvorkin / Deotte / cstdy / Gergely XGB-LGBM, Mizushima, Naji V1), OOF 0.946185
  pub_realmlp    Vladimir Demidov, RealMLP PyTorch, OOF 0.945871
  pub_mega_A..D  megayak, "four feature views" (kernel s6e9-four-feature-views-one-ensemble-lb-0-94639), 10-fold s42:
                 A = hybrid LGBM 0.946264, B = XGB on A's frame 0.946258, C = no digits + centred-window rates + lift, smooth 2/30/300 0.946223,
                 D = no exact key, //10/50/500/5000 ladder + windows, smooth auto/20/200 0.946077 (test columns joined on id)"""
import numpy as np, pandas as pd
from features import load
tr, te, o, y, feats = load(); P = "data/public/"
SRC = {"pub_megayak10": (np.load(P + "megayak/oof_hybrid.npy"), np.load(P + "megayak/test_hybrid.npy")),
       "pub_naji_v3": (pd.read_csv(P + "naji_v3/Pure LGBM_V3_oof.csv").OOF_Pred.values, pd.read_csv(P + "naji_v3/Pure LGBM_V3_test.csv").Will_Buy_EV.values),
       "pub_sergey": (pd.read_csv(P + "naji_v3/Sergey_LGBM_oof.csv").Will_Buy_EV.values, pd.read_csv(P + "naji_v3/Sergey_LGBM_submission.csv").Will_Buy_EV.values),
       "pub_naji_01blend": (pd.read_csv(P + "naji_v3/01_blend_oof.csv").Will_Buy_EV.values, pd.read_csv(P + "naji_v3/01_submission.csv").Will_Buy_EV.values),
       "pub_realmlp": (pd.read_csv(P + "realmlp/oof_preds.csv").Will_Buy_EV.values, pd.read_csv(P + "realmlp/submission.csv").Will_Buy_EV.values)}
m4o = pd.read_csv(P + "megayak4/oof_four_views.csv"); m4t = pd.read_csv(P + "megayak4/test_four_views.csv").set_index("id").loc[te.id]
assert (m4o.row.values == np.arange(len(tr))).all()
for v in "ABCD": SRC[f"pub_mega_{v}"] = (m4o[f"view_{v}"].values, m4t[f"view_{v}"].values)
SRC["pub_realmlp2"] = (pd.read_csv(P + "realmlp2/oof_preds.csv").Will_Buy_EV.values, pd.read_csv(P + "realmlp2/submission.csv").Will_Buy_EV.values)   # yekenot, version of 2026-09-13
SRC["pub_naji_xgb"] = (pd.read_csv(P + "naji_xgb/oof_XGBOOST.csv").OOF_Pred.values, pd.read_csv(P + "naji_xgb/test_XGBOOST.csv").Will_Buy_EV.values)   # Naji, XGB triple-TE + gain pruning, 10-fold s42, OOF 0.94624
for k, (a, b) in SRC.items():
    assert len(a) == len(tr) and len(b) == len(te), k
    np.save(f"submissions/oof_{k}.npy", a); pd.DataFrame({"id": te.id, "Will_Buy_EV": b}).to_csv(f"submissions/{k}.csv", index=False); print("wrote", k)
