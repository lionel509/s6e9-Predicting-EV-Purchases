"""Post-processing check for the 'deterministic boundary' rules from taeyangg4's 0.94649 notebook, on OUR blend_v7 OOF.
Rules (all measured on train): income >= 170537 -> all buyers; 31004 <= income <= 41970 -> none; commute >= 83 -> none;
income == 30000 & no subsidy & (concern == 1 | anxiety in Medium/High) -> none. If the blend already ranks these rows at the
extremes the shift is worth nothing; if not, the OOF says how much. Rules are re-derived on fit folds too (nested check)."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata
r = lambda a: rankdata(a) / len(a)
tr = pd.read_csv("data/train.csv"); y = (tr.Will_Buy_EV == "Yes").astype(int).values
W = {"hybrid10": (0.627, ["v27_hybrid_k10_s42", "v27_hybrid_k10_s7", "v27_hybrid_k10_s2026", "v27_hybrid_k10_s101", "v27_hybrid_k10_s202", "v27_hybrid_k10_s303", "v27_hybrid_k10_s404", "v27_hybrid_k10_s505"]),
     "naji_pc": (0.054, ["v22_pseudo_clean_naji_m1", "v22_pseudo_clean_naji_m1_f7_t0", "v22_pseudo_clean_naji_m1_f2026_t0"]),
     "pub_megayak10": (0.01, ["pub_megayak10"]), "pub_realmlp": (0.069, ["pub_realmlp"]), "pub_naji_01blend": (0.24, ["pub_naji_01blend"])}
blend = sum(w * np.mean([r(np.load(f"submissions/oof_{n}.npy")) for n in names], 0) for w, names in W.values()) / sum(w for w, _ in W.values())
hyb = np.mean([r(np.load(f"submissions/oof_{n}.npy")) for n in W["hybrid10"][1]], 0)
inc = tr.Annual_Income_USD.values; km = tr.Daily_Commute_km.values
masks = {"upper_cliff": inc >= 170537, "income_dead": (inc >= 31004) & (inc <= 41970), "commute_dead": km >= 83,
         "30k_zero": (inc == 30000) & (tr.Subsidy_Available == "No").values & ((tr.Environmental_Concern_Level == 1) | tr.Range_Anxiety_Level.isin(["Medium", "High"])).values}
sign = {"upper_cliff": +1, "income_dead": -1, "commute_dead": -1, "30k_zero": -1}
for nm, s in (("blend_v7", blend), ("hybrid10", hyb)):
    base = roc_auc_score(y, s); print(f"{nm}: OOF {base:.6f}")
    tot = s.copy()
    for k, m in masks.items():
        print(f"  {k:13s} rows {m.sum():6d}  buyers {y[m].sum():5d}  rate {y[m].mean():.4f}  score pct mean {s[m].mean():.3f} min {s[m].min():.3f} max {s[m].max():.3f}", end="")
        z = s.copy(); z[m] += 10 * sign[k]; print(f"  -> AUC {roc_auc_score(y, z):.6f} ({roc_auc_score(y, z)-base:+.6f})")
        tot[m] += 10 * sign[k]
    print(f"  all four rules: {roc_auc_score(y, tot):.6f} ({roc_auc_score(y, tot)-base:+.6f})")
# nested: derive every rule's support from the fit folds only (a rule counts only if it is pure on the fit rows)
cv = StratifiedKFold(5, shuffle=True, random_state=42); z = blend.copy(); base = roc_auc_score(y, blend)
for a, b in cv.split(tr, y):
    for k, m in masks.items():
        pure = (y[a][m[a]].mean() == (1 if sign[k] > 0 else 0)) if m[a].sum() else False
        if pure: z[b[m[b]]] += 10 * sign[k]
print(f"nested (rules kept only when pure on the fit folds): {roc_auc_score(y, z):.6f} ({roc_auc_score(y, z)-base:+.6f})")
