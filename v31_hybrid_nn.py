"""v31: v12's MLP (512/256/64, SiLU, embeddings for the six cats) on the v27 hybrid frame with the triple TE.
Numerics standardised on the fit rows; income and the two exact-value frequencies log1p'd. Best epoch by validation
AUC. MPS. The RealMLP public OOF (0.94587) earned 13% of blend_v4 — this is our own version of that slot.
Usage: python v31_hybrid_nn.py [epochs] [seed]"""
import sys, time, math, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from features import load
from v27_hybrid import build, fold_frames, CATS, TARGET
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 20; SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42; N = 5
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu"); torch.manual_seed(SEED); np.random.seed(SEED)
name = f"v31_hybrid_nn_s{SEED}"; rk = lambda v: rankdata(v) / len(v)
t0 = time.time(); tr, te, o, y, feats = load(); X, Xte, K, Kte = build(tr, te, o)
card = {c: len(X[c].cat.categories) for c in CATS}; LOGS = ["Annual_Income_USD", "fq_inc", "fq_km"]
ref = roc_auc_score(y, np.load("submissions/oof_v27_hybrid_k5_s42.npy")); print(f"{name}: v27 ref OOF {ref:.6f}  device {dev}", flush=True)

def prep(Xa, Xb, Xt):
    num = [c for c in Xa.columns if c not in CATS]
    A, B, T = (Xd[num].astype(float).copy() for Xd in (Xa, Xb, Xt))
    for c in LOGS: A[c], B[c], T[c] = np.log1p(A[c]), np.log1p(B[c]), np.log1p(T[c])
    mu, sd = A.mean(), A.std().replace(0, 1)
    f = lambda Xd: torch.tensor(((Xd - mu) / sd).fillna(0).values, dtype=torch.float32)
    g = lambda Xd: torch.tensor(np.column_stack([Xd[c].cat.codes.values for c in CATS]), dtype=torch.long)
    return (f(A), g(Xa)), (f(B), g(Xb)), (f(T), g(Xt))

class Net(nn.Module):
    def __init__(self, n_num):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(card[c], min(8, math.ceil(card[c] / 2) + 1)) for c in CATS])
        d = n_num + sum(e.embedding_dim for e in self.emb)
        self.mlp = nn.Sequential(nn.Linear(d, 512), nn.SiLU(), nn.Dropout(0.15), nn.Linear(512, 256), nn.SiLU(), nn.Dropout(0.15),
                                 nn.Linear(256, 64), nn.SiLU(), nn.Linear(64, 1))
    def forward(self, xn, xc): return self.mlp(torch.cat([xn] + [e(xc[:, i]) for i, e in enumerate(self.emb)], 1)).squeeze(1)

def predict(net, xn, xc, bs=65536):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(xn), bs): out.append(torch.sigmoid(net(xn[i:i+bs].to(dev), xc[i:i+bs].to(dev))).cpu())
    return torch.cat(out).numpy()

cv = StratifiedKFold(N, shuffle=True, random_state=42); oof = np.zeros(len(X)); pred = np.zeros(len(Xte)); BS = 2048
for k, (a, b) in enumerate(cv.split(X, y)):
    Xa, Xb, Xt = fold_frames(X, Xte, K, Kte, y, a, b); (na, ca), (nb, cb), (nt, ct) = prep(Xa, Xb, Xt)
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
    net.load_state_dict(best_state); oof[b] = predict(net, nb, cb); pred += rk(predict(net, nt, ct)) / N
    print(f"  fold {k} best {best:.6f} at epoch {int(np.argmax(hist))+1}  {hist}  {time.time()-t0:.0f}s", flush=True)
auc = roc_auc_score(y, oof); print(f"{name} OOF {auc:.6f}  Δ vs v27 {auc-ref:+.6f}  {time.time()-t0:.0f}s", flush=True)
np.save(f"submissions/oof_{name}.npy", oof); pd.DataFrame({"id": te.id, TARGET: pred}).to_csv(f"submissions/{name}.csv", index=False)
