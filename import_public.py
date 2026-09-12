"""Copy the public OOF / test pairs in data/public/ into the submissions/ convention (oof_<name>.npy + <name>.csv)
so assemble.py can use them as groups. Names are prefixed pub_ so their provenance is visible in every blend line.
Sources (all on StratifiedKFold(5, shuffle, 42) or a strict superset partition — see the hub note):
  pub_megayak10  megayak, one LightGBM from raw data, 10-fold, OOF 0.946264 (fold-ranked, ES on the held-out fold)
  pub_naji_v3    Naji, Pure LGBM V3, OOF 0.946064
  pub_sergey     Sergey, focal-loss LGBM, OOF 0.945329 (logits)
  pub_realmlp    Vladimir Demidov, RealMLP PyTorch, OOF 0.945871"""
import numpy as np, pandas as pd
from features import load
tr, te, o, y, feats = load(); P = "data/public/"
SRC = {"pub_megayak10": (np.load(P + "megayak/oof_hybrid.npy"), np.load(P + "megayak/test_hybrid.npy")),
       "pub_naji_v3": (pd.read_csv(P + "naji_v3/Pure LGBM_V3_oof.csv").OOF_Pred.values, pd.read_csv(P + "naji_v3/Pure LGBM_V3_test.csv").Will_Buy_EV.values),
       "pub_sergey": (pd.read_csv(P + "naji_v3/Sergey_LGBM_oof.csv").Will_Buy_EV.values, pd.read_csv(P + "naji_v3/Sergey_LGBM_submission.csv").Will_Buy_EV.values),
       "pub_realmlp": (pd.read_csv(P + "realmlp/oof_preds.csv").Will_Buy_EV.values, pd.read_csv(P + "realmlp/submission.csv").Will_Buy_EV.values)}
for k, (a, b) in SRC.items():
    assert len(a) == len(tr) and len(b) == len(te), k
    np.save(f"submissions/oof_{k}.npy", a); pd.DataFrame({"id": te.id, "Will_Buy_EV": b}).to_csv(f"submissions/{k}.csv", index=False); print("wrote", k)
