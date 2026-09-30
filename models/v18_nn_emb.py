"""v18: MLP with learned embeddings for income (13k levels), commute (805) and age (45) — a different estimator of
the per-value effect than target encoding, for blend diversity. No TE columns; Base counts / original lookups kept.
Vocabulary per fold from the fit rows; unseen values -> index 0. Usage: python v18_nn_emb.py [epochs] [seed]"""
import sys, time, math, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from features import *
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 20; SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu"); torch.manual_seed(SEED); np.random.seed(SEED)
tr, te, o, y, feats = load(); base = Base(tr, te, o, feats); Btr, Bte = base(tr), base(te); skf = folds(tr, y)
ref = roc_auc_score(y, np.load("submissions/oof_v6b_round.npy")); print(f"v6b ref OOF {ref:.6f}  device {dev}", flush=True)
cats = base.cats; card = {c: len(base.cat_levels[c]) for c in cats}
HI = {"Annual_Income_USD": 24, "Daily_Commute_km": 12, "Age": 6}   # embedding dims for the high-cardinality numerics
LOGS = ["Annual_Income_USD", "Annual_Income_USD_cnt", "Daily_Commute_km_cnt", "inc_orig_cnt", "com_orig_cnt"]

def prep(Xa, Xb, Xt, dfa, dfb, dft):
    num = [c for c in Xa.columns if c not in cats]
    A, B, T = (X[num].astype(float).copy() for X in (Xa, Xb, Xt))
    for c in LOGS: A[c], B[c], T[c] = np.log1p(A[c]), np.log1p(B[c]), np.log1p(T[c])
    for c in [c for c in num if A[c].isna().any() or B[c].isna().any() or T[c].isna().any()]:
        for X in (A, B, T): X[c + "_nan"] = X[c].isna().astype(float)
        mu = A[c].mean(); A[c], B[c], T[c] = A[c].fillna(mu), B[c].fillna(mu), T[c].fillna(mu)
    mu, sd = A.mean(), A.std().replace(0, 1)
    f = lambda X: torch.tensor(((X - mu) / sd).values, dtype=torch.float32)
    vocab = {c: {v: i + 1 for i, v in enumerate(pd.unique(dfa[c]))} for c in HI}
    g = lambda X, df: torch.tensor(np.column_stack([X[c].cat.codes.values for c in cats] + [df[c].map(vocab[c]).fillna(0).astype(int).values for c in HI]), dtype=torch.long)
    sizes = [card[c] for c in cats] + [len(vocab[c]) + 1 for c in HI]; dims = [min(8, math.ceil(card[c] / 2) + 1) for c in cats] + [HI[c] for c in HI]
    return (f(A), g(Xa, dfa)), (f(B), g(Xb, dfb)), (f(T), g(Xt, dft)), sizes, dims

class Net(nn.Module):
    def __init__(self, n_num, sizes, dims):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(s, d) for s, d in zip(sizes, dims)]); self.drop_e = nn.Dropout(0.1)
        d = n_num + sum(dims)
        self.mlp = nn.Sequential(nn.Linear(d, 512), nn.SiLU(), nn.Dropout(0.2), nn.Linear(512, 256), nn.SiLU(), nn.Dropout(0.2), nn.Linear(256, 64), nn.SiLU(), nn.Linear(64, 1))
    def forward(self, xn, xc):
        return self.mlp(torch.cat([xn, self.drop_e(torch.cat([e(xc[:, i]) for i, e in enumerate(self.emb)], 1))], 1)).squeeze(1)

def predict(net, xn, xc, bs=65536):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(xn), bs): out.append(torch.sigmoid(net(xn[i:i+bs].to(dev), xc[i:i+bs].to(dev))).cpu())
    return torch.cat(out).numpy()

oof = np.zeros(len(tr)); pred = np.zeros(len(te)); t0 = time.time(); BS = 2048
for k, (a, b) in enumerate(skf):
    (na, ca), (nb, cb), (nt, ct), sizes, dims = prep(Btr.iloc[a], Btr.iloc[b], Bte, tr.iloc[a], tr.iloc[b], te)
    ya = torch.tensor(y[a], dtype=torch.float32); na, ca, ya = na.to(dev), ca.to(dev), ya.to(dev)
    net = Net(na.shape[1], sizes, dims).to(dev)
    emb_params = [p for e in net.emb for p in e.parameters()]; other = [p for n_, p in net.named_parameters() if not n_.startswith("emb")]
    opt = torch.optim.AdamW([{"params": other, "weight_decay": 1e-5}, {"params": emb_params, "weight_decay": 1e-3}], lr=1e-3)
    steps = EPOCHS * math.ceil(len(a) / BS); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.2)
    lossf = nn.BCEWithLogitsLoss(); best, best_state, hist = -1, None, []
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(len(a), device=dev)
        for i in range(0, len(a), BS):
            idx = perm[i:i+BS]; opt.zero_grad(); loss = lossf(net(na[idx], ca[idx]), ya[idx]); loss.backward(); opt.step(); sched.step()
        auc = roc_auc_score(y[b], predict(net, nb, cb)); hist.append(round(auc, 5))
        if auc > best: best, best_state = auc, {k_: v.detach().clone() for k_, v in net.state_dict().items()}
    net.load_state_dict(best_state); oof[b] = predict(net, nb, cb); pred += predict(net, nt, ct) / len(skf)
    print(f"  fold {k} best {best:.6f} at epoch {int(np.argmax(hist))+1}  {hist[-6:]}  {time.time()-t0:.0f}s", flush=True)
auc = roc_auc_score(y, oof); name = f"v18_nn_emb_s{SEED}"
print(f"{name:12s} OOF {auc:.6f}  {time.time()-t0:.0f}s\n   Δ vs v6b {auc-ref:+.6f}", flush=True)
np.save(f"submissions/oof_{name}.npy", oof); pd.DataFrame({"id": te.id, "Will_Buy_EV": pred}).to_csv(f"submissions/{name}.csv", index=False)
