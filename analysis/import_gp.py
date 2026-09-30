"""Import the public OOF engines lucifer19's EV Grand Prix v7 blends (2026-09-28) that import_public.py does not already
cover, into the submissions/ convention as gp_<name> (oof_gp_<name>.npy + gp_<name>.csv). Loader logic ported from
that notebook's Lap 2; sources downloaded to data/public/gp/ (datasets + kernel outputs, see issue #1)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "models"))   # models/ holds features + the v* scripts
import numpy as np, pandas as pd
from pathlib import Path
from features import load
tr, te, o, y, feats = load(); ROOT = Path("data/public/gp"); ID, T = "id", "Will_Buy_EV"
tr_ids, te_ids = tr.id.values, te.id.values
def find(pat, hint=None):
    hits = sorted(p for p in ROOT.rglob(pat) if p.is_file() and (hint is None or hint in str(p).lower()))
    assert hits, (pat, hint); return hits[0]
al = lambda f, ids, c: f.set_index(ID)[c].reindex(ids).to_numpy(np.float64)
S = {}
f = pd.read_csv(find("oof_CATBOOST.csv", "catboost-triple-te")); S["nj_catboost"] = (al(f, tr_ids, "OOF_Pred"), al(pd.read_csv(find("test_CATBOOST.csv", "catboost-triple-te")), te_ids, T))
gd = find("folds_seed42.npy", "golem").parent
for p in sorted(gd.glob("oof_*.npy")): S["gl_" + p.stem[4:]] = (np.load(p), np.load(gd / p.name.replace("oof_", "test_", 1)))
fo, ft = pd.read_csv(find("oof_components.csv", "fusion-iv")), pd.read_csv(find("test_components.csv", "fusion-iv"))
for c in [c for c in ft.columns if c != ID]: S["fx_" + c] = (al(fo, tr_ids, c), al(ft, te_ids, c))
S["bx_medvax_blamerx"] = (al(pd.read_csv(find("oof.csv", "medvax")), tr_ids, "pred"), al(pd.read_csv(find("submission.csv", "medvax")), te_ids, T))
S["bl_blamerx"] = (al(pd.read_csv(find("oof.csv", "window-encodings")), tr_ids, "pred"), al(pd.read_csv(find("submission.csv", "window-encodings")), te_ids, T))
for pre, hint, members in (("dl_", "digit-leak", ("lgb", "cat", "xgb")),
                           ("tm_", "hirge", ("cat_native", "conditional_moe", "ft_transformer", "gated_crossnet", "hirge", "lgb_raw", "lgb_te", "offset_residual", "tabm_lite", "tabm_te", "xgb_freq", "xgb_te"))):
    d = find(f"oof_{members[0]}.npy", hint).parent
    for m in members: S[pre + m] = (np.load(d / f"oof_{m}.npy"), np.load(d / f"test_{m}.npy"))
try:
    d = find("oof_xgb_fp.npy").parent
    for m in ("xgb_fp", "xgb_fp_hb", "xgb_fp_kern", "logit_fp", "logit_fp_hb", "nn_fp_wide"): S["tf_" + m] = (np.load(d / f"oof_{m}.npy"), np.load(d / f"test_{m}.npy"))
except AssertionError: print("tf_ source not downloaded, skipped")
S["ev_xgb"] = (np.load(find("oof_preds_base.npy")), np.load(find("test_preds_base.npy")))
S["kr_lgb"] = (np.load(find("oof_predictions_lgb_42_5.npy")), np.load(find("test_predictions_lgb_42_5.npy")))
S["hj_xgb"] = (al(pd.read_parquet(find("XGB_SAMPLE_OOF.parquet")), tr_ids, "oof_pred"), al(pd.read_parquet(find("XGB_SAMPLE_TEST.parquet")), te_ids, "test_pred"))
for k, h in (("mz_lgb_comp", "competition-only"), ("mz_lgb_tfe", "leak-free")): S[k] = (al(pd.read_csv(find("oof_predictions.csv", h)), tr_ids, T), al(pd.read_csv(find("test_predictions.csv", h)), te_ids, T))
S["mm_memory"] = (al(pd.read_csv(find("oof_generator_memory.csv")), tr_ids, "oof"), al(pd.read_csv(find("submission.csv", "generator-remembers")), te_ids, T))
S["ak_histgbm"] = (al(pd.read_csv(find("oof_proba.csv", "signal-that-matters")), tr_ids, T), al(pd.read_csv(find("test_proba.csv", "signal-that-matters")), te_ids, T))
S["yk_realmlp_ptk"] = (al(pd.read_csv(find("oof_preds.csv", "realmlp-pytabkit")), tr_ids, T), al(pd.read_csv(find("submission.csv", "realmlp-pytabkit")), te_ids, T))
for k, (a, b) in S.items():
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    assert len(a) == len(tr) and len(b) == len(te) and np.isfinite(a).all() and np.isfinite(b).all(), k
    np.save(f"submissions/oof_gp_{k}.npy", a); pd.DataFrame({ID: te_ids, T: b}).to_csv(f"submissions/gp_{k}.csv", index=False)
print(len(S), "engines written:", " ".join(S))
