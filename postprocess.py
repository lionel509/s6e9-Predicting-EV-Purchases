"""Apply the four deterministic boundary rules (pp_rules.py: +0.000005 OOF on blend_v7, nested check identical) to a submission's
test column in rank space: rows the generator never lets buy go to the bottom, rows it always makes buy go to the top.
Rules (train, 668,665 rows): income >= 170537 -> 393/393 buyers; 31004 <= income <= 41970 -> 0/1257; commute >= 83 -> 0/186;
income == 30000 & no subsidy & (concern == 1 | anxiety Medium/High) -> 0/7157.   Usage: python postprocess.py <name>  -> <name>_pp.csv"""
import sys, numpy as np, pandas as pd
name = sys.argv[1]; te = pd.read_csv("data/test.csv"); sub = pd.read_csv(f"submissions/{name}.csv")
assert (sub.id.values == te.id.values).all()
inc = te.Annual_Income_USD.values; km = te.Daily_Commute_km.values; p = sub.Will_Buy_EV.values.astype(float)
masks = {"upper_cliff": (inc >= 170537, +10.0), "income_dead": ((inc >= 31004) & (inc <= 41970), -10.0), "commute_dead": (km >= 83, -5.0),
         "30k_zero": ((inc == 30000) & (te.Subsidy_Available == "No").values & ((te.Environmental_Concern_Level == 1) | te.Range_Anxiety_Level.isin(["Medium", "High"])).values, -5.0)}
q = p.copy()
for k, (m, s) in masks.items(): q[m] += s; print(f"  {k:13s} {m.sum():6d} test rows  (mean rank before {p[m].mean():.3f})")
from scipy.stats import rankdata
q = rankdata(q) / len(q); pd.DataFrame({"id": te.id, "Will_Buy_EV": q}).to_csv(f"submissions/{name}_pp.csv", index=False)
print(f"wrote submissions/{name}_pp.csv  rows moved {sum(m.sum() for m, _ in masks.values())}")
