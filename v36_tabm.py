"""v36: TabM (yandex-research/tabm 0.0.3, "TabM: Advancing Tabular Deep Learning with Parameter-Efficient
Ensembling") on Apple MPS — a second neural member on exactly v35's inputs, so it is decorrelated from the
LightGBM family by architecture rather than by features.
Inputs are v35's RealMLP inputs verbatim (nn_features.feature_engineering + NumericalPreprocessor + the
per-fold nested TargetEncoder over the three combo keys). Numerics go through piecewise-linear embeddings
(rtdl_num_embeddings.compute_bins on the FIT rows only + PiecewiseLinearEmbeddings version="B"); every
categorical gets its own nn.Embedding (one-hot is out: the 24 cat columns sum to ~40k categories).
Model: EnsembleView -> make_tabm_backbone(arch_type="tabm", k=32, 3 blocks x 512, dropout 0.1) ->
LinearEnsemble head, i.e. the package README's "custom inputs" composition. Loss is the MEAN of the k
submodels' BCE losses (not the loss of the mean); prediction is the MEAN of the k probabilities.
Per fold: validation AUC every epoch, the best epoch's validation probabilities become the OOF and its
weights predict test — the same "early selection" v35 does with best_val_probs_.
Usage: python v36_tabm.py [n_folds=5] [seed=42] [epochs=12]     SMOKE=1: 40k rows, 1 epoch, no files
"""
import os, sys, time, json, warnings
SMOKE = os.environ.get("SMOKE") == "1"
N_ = int(sys.argv[1]) if len(sys.argv) > 1 else 5
SEED_ = int(sys.argv[2]) if len(sys.argv) > 2 else 42
EPOCHS_DEFAULT = 12
EPOCHS_ = int(sys.argv[3]) if len(sys.argv) > 3 else EPOCHS_DEFAULT
import random
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata
from rtdl_num_embeddings import compute_bins, PiecewiseLinearEmbeddings
from tabm import EnsembleView, LinearEnsemble, make_tabm_backbone
import nn_features as nf

warnings.filterwarnings("ignore")
rk = lambda v: rankdata(v) / len(v)
name = f"v36_tabm_k{N_}_s{SEED_}" + ("" if EPOCHS_ == EPOCHS_DEFAULT else f"_e{EPOCHS_}") + ("_smoke" if SMOKE else "")

# ── config (TabM defaults from the package README; batch raised for 600k rows on MPS) ──
K_ENS, N_BLOCKS, D_BLOCK, DROPOUT = 32, 3, 512, 0.1
LR, WEIGHT_DECAY = 2e-3, 3e-4
BATCH, EVAL_BATCH = 1024, 16384
D_NUM_EMB, N_BINS, BIN_SAMPLE = 16, 48, 200_000
DEVICE = torch.device("mps")

def seed_everything(seed: int):
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)

def cat_emb_dim(card: int) -> int:
    return int(min(16, max(2, np.ceil(np.sqrt(card)))))

class TabMNet(nn.Module):
    """The README's custom-input composition: per-feature input modules -> EnsembleView -> TabM backbone
    -> LinearEnsemble. The k representations are created before the first linear layer mixes features."""
    def __init__(self, bins, cat_cards, k=K_ENS, n_blocks=N_BLOCKS, d_block=D_BLOCK, dropout=DROPOUT, d_num=D_NUM_EMB):
        super().__init__()
        self.num_module = PiecewiseLinearEmbeddings(bins, d_num, activation=False, version="B")
        self.cat_dims = [cat_emb_dim(c) for c in cat_cards]
        self.cat_modules = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(cat_cards, self.cat_dims)])
        d_features = [d_num] * len(bins) + self.cat_dims
        self.ensemble_view = EnsembleView(k=k)
        self.backbone = make_tabm_backbone(d_in=sum(d_features), n_blocks=n_blocks, d_block=d_block, dropout=dropout,
                                           k=k, arch_type="tabm", start_scaling_init="normal",
                                           start_scaling_init_chunks=d_features)
        self.output = LinearEnsemble(self.backbone.get_original_output_shape()[0], 1, k=k)
        self.d_in = sum(d_features)
    def forward(self, x_num, x_cat):
        parts = [self.num_module(x_num).flatten(1, -1)]
        parts += [m(x_cat[:, i]) for i, m in enumerate(self.cat_modules)]
        x = torch.column_stack(parts)
        return self.output(self.backbone(self.ensemble_view(x)))   # (B, k, 1)

@torch.no_grad()
def predict(model, xn, xc):
    """Mean of the k submodels' PROBABILITIES (README: average probabilities, not logits)."""
    model.eval()
    return np.concatenate([torch.sigmoid(model(xn[s:s + EVAL_BATCH], xc[s:s + EVAL_BATCH])).mean(1).squeeze(-1).float().cpu().numpy()
                           for s in range(0, len(xn), EVAL_BATCH)])

if __name__ == "__main__":
    t0 = time.time(); seed_everything(SEED_)
    X, y, X_test, test_id, cat_cols, num_cols, combo_names = nf.load_frames(smoke_rows=40000 if SMOKE else None)
    epochs = 1 if SMOKE else EPOCHS_
    print(f"{name}: {len(X)} rows, {X.shape[1]} features ({len(cat_cols)} cat / {len(num_cols)} num), "
          f"k={K_ENS} {N_BLOCKS}x{D_BLOCK} drop {DROPOUT} lr {LR} wd {WEIGHT_DECAY} bs {BATCH} epochs {epochs} "
          f"device {DEVICE.type}  ({time.time()-t0:.0f}s)", flush=True)
    skf = StratifiedKFold(n_splits=N_, shuffle=True, random_state=SEED_)
    oof = np.zeros(len(X)); pte = np.zeros(len(X_test)); scores = []; best_epochs = []
    for fold, (a, b) in enumerate(skf.split(X, y)):
        seed_everything(SEED_ + fold)
        X_tr, X_val, X_tst = X.iloc[a].copy(), X.iloc[b].copy(), X_test.copy()
        y_tr, y_val = y.iloc[a].to_numpy(), y.iloc[b].to_numpy()
        nf.fold_te(X_tr, X_val, X_tst, y_tr, combo_names, SEED_)
        num_names = [c for c in X_tr.columns if c not in cat_cols]
        # ── numerics: v35's NumericalPreprocessor fitted on the fit rows only ──
        tr_num = X_tr[num_names].values.astype(np.float32)
        prep = nf.NumericalPreprocessor(nf.TFMS); prep.fit(tr_num)
        tr_num = prep.transform(tr_num)
        va_num = prep.transform(X_val[num_names].values.astype(np.float32))
        te_num = prep.transform(X_tst[num_names].values.astype(np.float32))
        keep = np.array([len(np.unique(tr_num[:, i])) > 1 for i in range(tr_num.shape[1])])
        tr_num, va_num, te_num = tr_num[:, keep], va_num[:, keep], te_num[:, keep]
        # ── categoricals: integer codes clipped into each embedding's range (v35 clips the same way) ──
        tr_cat = X_tr[cat_cols].values.astype(np.int64); va_cat = X_val[cat_cols].values.astype(np.int64)
        te_cat = X_tst[cat_cols].values.astype(np.int64)
        cards = (np.concatenate([tr_cat, va_cat, te_cat], 0).max(0) + 1).tolist()
        cmax = np.array(cards) - 1
        tr_cat, va_cat, te_cat = np.clip(tr_cat, 0, cmax), np.clip(va_cat, 0, cmax), np.clip(te_cat, 0, cmax)
        # ── piecewise-linear bins from the FIT rows only ──
        sub = tr_num if len(tr_num) <= BIN_SAMPLE else tr_num[np.random.RandomState(SEED_).choice(len(tr_num), BIN_SAMPLE, replace=False)]
        bins = compute_bins(torch.as_tensor(sub, dtype=torch.float32), n_bins=min(N_BINS, len(sub) - 1))
        bins = [x.to(DEVICE) for x in bins]
        Xtn = torch.as_tensor(tr_num, device=DEVICE); Xtc = torch.as_tensor(tr_cat, device=DEVICE)
        Xvn = torch.as_tensor(va_num, device=DEVICE); Xvc = torch.as_tensor(va_cat, device=DEVICE)
        Xsn = torch.as_tensor(te_num, device=DEVICE); Xsc = torch.as_tensor(te_cat, device=DEVICE)
        ytt = torch.as_tensor(y_tr, dtype=torch.float32, device=DEVICE)
        model = TabMNet(bins, cards).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        lossf = nn.BCEWithLogitsLoss()
        if fold == 0:
            print(f"  d_in {model.d_in} ({len(bins)} num x {D_NUM_EMB} + cat dims {model.cat_dims})  "
                  f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
        best_auc, best_ep, best_val, best_state = -np.inf, 0, None, None
        order = np.arange(len(y_tr)); rng = np.random.RandomState(SEED_ + fold)
        for ep in range(epochs):
            model.train(); rng.shuffle(order)
            idx = torch.as_tensor(order, device=DEVICE)
            for s in range(0, len(order), BATCH):
                j = idx[s:s + BATCH]
                opt.zero_grad(set_to_none=True)
                logits = model(Xtn[j], Xtc[j]).squeeze(-1)                     # (bs, k)
                loss = lossf(logits, ytt[j].unsqueeze(1).expand_as(logits))    # mean over rows AND submodels
                loss.backward(); opt.step()
            pv = predict(model, Xvn, Xvc); sc = roc_auc_score(y_val, pv)
            if sc > best_auc:
                best_auc, best_ep, best_val = sc, ep + 1, pv
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"    epoch {ep+1}/{epochs}  val {sc:.6f}  best {best_auc:.6f}{'  *' if best_ep == ep + 1 else ''}  {time.time()-t0:.0f}s", flush=True)
        model.load_state_dict(best_state)
        oof[b] = best_val; pte += rk(predict(model, Xsn, Xsc)) / N_
        scores.append(best_auc); best_epochs.append(best_ep)
        print(f"  fold {fold}: auc {best_auc:.6f}  best epoch {best_ep}  {time.time()-t0:.0f}s", flush=True)
        del model, best_state, Xtn, Xtc, Xvn, Xvc, Xsn, Xsc, ytt
        torch.mps.empty_cache()
    auc = roc_auc_score(y, oof)
    print(f"{name} OOF {auc:.6f}  folds {[round(s, 5) for s in scores]}  best epochs {best_epochs}  {time.time()-t0:.0f}s", flush=True)
    if SMOKE: sys.exit(0)
    np.save(f"submissions/oof_{name}.npy", oof)
    pd.DataFrame({nf.ID: test_id, nf.TARGET: pte}).to_csv(f"submissions/{name}.csv", index=False)
    json.dump({"oof_auc": auc, "folds": scores, "iters": best_epochs, "n_splits": N_, "seed": SEED_, "epochs": epochs},
              open(f"submissions/{name}.json", "w"), indent=2)
