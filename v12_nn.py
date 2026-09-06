"""v12: an MLP on the v6b frame (Base + nested TE), for blend diversity — the S5E12 DNN half. Same folds as
v6b; categoricals as embeddings, numerics standardised on the fit rows, skewed counts log1p'd, NaN → mean + flag.
Best epoch by validation AUC. Runs on MPS if present. Usage: python v12_nn.py [epochs] [seed]"""
import sys, time, math, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from features import *
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 12; SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu"); torch.manual_seed(SEED); np.random.seed(SEED)
tr, te, o, y, feats = load(); base = Base(tr, te, o, feats); Btr, Bte = base(tr), base(te); skf = folds(tr, y)
ref = roc_auc_score(y, np.load("submissions/oof_v6b_round.npy")); print(f"v6b ref OOF {ref:.6f}  device {dev}", flush=True)
cats = base.cats; card = {c: len(base.cat_levels[c]) for c in cats}
LOGS = ["Annual_Income_USD", "Annual_Income_USD_cnt", "Daily_Commute_km_cnt", "inc_orig_cnt", "com_orig_cnt"]

def prep(Xa, Xb, Xt):
    num = [c for c in Xa.columns if c not in cats]
    A, B, T = (X[num].astype(float).copy() for X in (Xa, Xb, Xt))
    for c in LOGS:
        if c in num: A[c], B[c], T[c] = np.log1p(A[c]), np.log1p(B[c]), np.log1p(T[c])
    nan_cols = [c for c in num if A[c].isna().any() or B[c].isna().any() or T[c].isna().any()]
    for c in nan_cols:
        for X in (A, B, T): X[c + "_nan"] = X[c].isna().astype(float)
        mu = A[c].mean(); A[c], B[c], T[c] = A[c].fillna(mu), B[c].fillna(mu), T[c].fillna(mu)
    mu, sd = A.mean(), A.std().replace(0, 1)
    f = lambda X: torch.tensor(((X - mu) / sd).values, dtype=torch.float32)
    g = lambda X: torch.tensor(np.column_stack([X[c].cat.codes.values for c in cats]), dtype=torch.long)
    return (f(A), g(Xa)), (f(B), g(Xb)), (f(T), g(Xt))

class Net(nn.Module):
    def __init__(self, n_num):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(card[c], min(8, math.ceil(card[c] / 2) + 1)) for c in cats])
        d = n_num + sum(e.embedding_dim for e in self.emb)
        self.mlp = nn.Sequential(nn.Linear(d, 512), nn.SiLU(), nn.Dropout(0.15), nn.Linear(512, 256), nn.SiLU(), nn.Dropout(0.15),
                                 nn.Linear(256, 64), nn.SiLU(), nn.Linear(64, 1))
    def forward(self, xn, xc):
        return self.mlp(torch.cat([xn] + [e(xc[:, i]) for i, e in enumerate(self.emb)], 1)).squeeze(1)

def predict(net, xn, xc, bs=65536):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(xn), bs): out.append(torch.sigmoid(net(xn[i:i+bs].to(dev), xc[i:i+bs].to(dev))).cpu())
    return torch.cat(out).numpy()

tenc = TE(tr, te, V5_SPECS + ROUND_SPECS); oof = np.zeros(len(tr)); pred = np.zeros(len(te)); t0 = time.time(); BS = 2048
for k, (a, b) in enumerate(skf):
    Xa, Xb, Xt = tenc.fold(a, b, y, Btr, Bte); (na, ca), (nb, cb), (nt, ct) = prep(Xa, Xb, Xt)
    ya = torch.tensor(y[a], dtype=torch.float32); na, ca, ya = na.to(dev), ca.to(dev), ya.to(dev)
    net = Net(na.shape[1]).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    steps = EPOCHS * math.ceil(len(a) / BS); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.2)
    lossf = nn.BCEWithLogitsLoss(); best, best_state, hist = -1, None, []
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(len(a), device=dev)
        for i in range(0, len(a), BS):
            idx = perm[i:i+BS]; opt.zero_grad(); loss = lossf(net(na[idx], ca[idx]), ya[idx]); loss.backward(); opt.step(); sched.step()
        auc = roc_auc_score(y[b], predict(net, nb, cb)); hist.append(round(auc, 5))
        if auc > best: best, best_state = auc, {k_: v.detach().clone() for k_, v in net.state_dict().items()}
    net.load_state_dict(best_state); oof[b] = predict(net, nb, cb); pred += predict(net, nt, ct) / len(skf)
    print(f"  fold {k} best {best:.6f} at epoch {int(np.argmax(hist))+1}  {hist}  {time.time()-t0:.0f}s", flush=True)
auc = roc_auc_score(y, oof); name = f"v12_nn_s{SEED}"
print(f"{name:12s} OOF {auc:.6f}  {time.time()-t0:.0f}s\n   Δ vs v6b {auc-ref:+.6f}", flush=True)
np.save(f"submissions/oof_{name}.npy", oof); pd.DataFrame({"id": te.id, "Will_Buy_EV": pred}).to_csv(f"submissions/{name}.csv", index=False)
