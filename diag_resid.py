"""diag_resid: what signal is left in the best blend's OOF, by group key (Lionel's method step: residuals grouped by every key).
For each candidate key: calibrate the blend (logistic on its score), then a cross-fitted one-step Newton correction per group,
delta_g = sum(y - p) / (sum p(1-p) + M), learned on 4/5 of rows and applied to the other 1/5. Prints the AUC gain.
A gain near 0 means the blend already has that grouping; >= +0.00003 is worth a model run. The null row (random groups of
the same size as income tokens) sets the noise floor. Usage: python diag_resid.py [blend_name] [M]   (issue #1)"""
import sys, numpy as np, pandas as pd, tiktoken
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from features import load

NAME = sys.argv[1] if len(sys.argv) > 1 else "blend_v37fr_probit"; M = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
tr, te, o, y, feats = load()
s = np.load(f"submissions/oof_{NAME}.npy").astype(float)
z = LogisticRegression(C=1e6).fit(s[:, None], y).decision_function(s[:, None]); p = expit(z)
base = roc_auc_score(y, z); print(f"{NAME}: OOF {base:.6f}  (M={M:g})", flush=True)

enc = tiktoken.get_encoding("gpt2")
def toks(v):
    u, inv = np.unique(v, return_inverse=True); t = [enc.encode(" " + x) for x in u]
    return (np.array([str(a[0]) for a in t])[inv], np.array([str(a[-1]) + f"_{len(a)}" for a in t])[inv], np.array(["_".join(map(str, a[:2])) for a in t])[inv])

inc = tr.Annual_Income_USD.to_numpy(np.int64); km = tr.Daily_Commute_km.to_numpy(float)
inc_s = inc.astype(str); km_s = np.array([f"{v:g}" for v in km]); km_s1 = np.array([f"{v:.1f}" for v in km])
it1, itl, it2 = toks(inc_s); kt1, ktl, kt2 = toks(km_s)
CTX = ["Age", "Number_of_Cars_Owned", "Charging_Stations_Near_Home", "Charging_Stations_Near_Work", "Environmental_Concern_Level",
       "Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]
col = {c: tr[c].astype(str).to_numpy() for c in CTX}
x = lambda *a: np.char.add(np.char.add(a[0].astype(str), "|"), a[1].astype(str)) if len(a) == 2 else x(x(*a[:2]), *a[2:])

keys = {"null (random, |inc tok1| groups)": np.random.default_rng(0).integers(0, len(np.unique(it1)), len(y)).astype(str),
        "km exact": km_s, "km tok1": kt1, "km toklast": ktl, "km tok2": kt2, "km as %.1f text != %g": (km_s != km_s1).astype(str),
        "inc exact": inc_s, "inc tok1": it1, "inc toklast": itl, "inc tok2": it2, "inc exact x km exact": x(inc_s, km_s)}
for c in CTX:
    keys[f"inc tok1 x {c}"] = x(it1, col[c]); keys[f"inc toklast x {c}"] = x(itl, col[c]); keys[f"km exact x {c}"] = x(km_s, col[c])
for i, a in enumerate(CTX):
    for b in CTX[i + 1:]: keys[f"{a} x {b}"] = x(col[a], col[b])
keys["all categoricals"] = x(*[col[c] for c in CTX[5:]])

F = list(StratifiedKFold(5, shuffle=True, random_state=11).split(z, y)); r, h = y - p, p * (1 - p)
rows = []
for name, k in keys.items():
    codes = pd.factorize(k)[0]; n = codes.max() + 1; zz = z.copy()
    for a, b in F:
        num = np.bincount(codes[a], r[a], n); den = np.bincount(codes[a], h[a], n) + M
        zz[b] = z[b] + (num / den)[codes[b]]
    g = roc_auc_score(y, zz) - base; rows.append((g, name, n)); print(f"  {g:+.6f}  {name}  ({n} groups)", flush=True)
print("\ntop 15:"); [print(f"  {g:+.6f}  {nm}  ({n})") for g, nm, n in sorted(rows, reverse=True)[:15]]
