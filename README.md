# Kaggle Playground S6E9 — Predicting EV Purchases

Binary classification, ROC-AUC, 2026-09-01 → 09-30. **The record is the vault note**
`Citadel/Active/Kaggle S6E9 — Predicting EV Purchases.md` (run log, standing, traps, next step); this file is the map of the repo.

## Setup
```
uv sync --reinstall                                   # after any directory move, or console scripts point at the old path
uv run kaggle competitions download -c playground-series-s6e9 -p data
uv run kaggle datasets download itzzomkar/ev-adoption-behavior-and-range-anxiety -p data/orig --unzip
python import_public.py                               # after fetching the public OOF files listed in its docstring into data/public/
```
Auth is a single token at `~/.kaggle/access_token` (CLI ≥ 2). Handle `lionelw509`, team `lionel W509`.

## Layout
- `features.py` — shared pipeline for the v2–v26 line: `Base` (cats, counts, original lookup) + nested `TE(m)` + `run_cv`. Folds are `StratifiedKFold(5, shuffle=True, random_state=42)` everywhere; public notebooks use the same split, so OOFs are stack-compatible.
- `v27_hybrid.py` — the current frame: megayak's recipe (Naji V3 triple TE over 15 keys incl. Smooth Keys, digit / modulo / quantisation ladder, flags, original means) on our split. `build()` and `fold_frames()` are imported by v28–v31. Args: folds, seed.
- `v28_hybrid_plus.py` (our m=1 TE / clean pseudo on top), `v29_hybrid_cat.py`, `v30_hybrid_xgb.py`, `v31_hybrid_nn.py` — levers and other families on the hybrid frame. All measured flat or worse; see the note.
- `assemble.py <out>` — the blend. Each entry in `GROUPS` is a list of OOF names rank-averaged into one signal plus an optional test-column override; weights by coordinate search over the simplex + Nelder-Mead polish (plain Nelder-Mead stalls past ~10 groups). Writes `submissions/<out>.csv` and `oof_<out>.npy`.
- `stack.py` — nested LR / LightGBM stackers over the group columns; loses to the linear blend.
- `diag_public_oof.py` — AUC, Spearman and nested blend fits of the public OOF files against ours.
- `blend.py` — the older two-to-five-model rank blend; `tune.py` + `tuning.db` — the 09-01 Optuna study (pre-TE frame, stale).
- `run_chain*.sh` — sequential CPU queues; every line of console output is appended to `runs.log`.
- `submissions/` (gitignored) — `oof_<name>.npy` + `<name>.csv` for every run; `pub_*` are the public sources. `leaderboard/` — dated board snapshots.

## Rules of the repo
- Every run gets a row in the note's run log, including the ones that went nowhere.
- Nothing goes to the board without a leak-free OOF above the incumbent blend's. Refits and seed bags have no OOF and ride inside a blend.
- Pseudo-labels come only from a model that never saw the validation fold (`v22`, `v28 --pseudo`); un-nested TE and fold-averaged soft labels both inflate OOF by 0.0002–0.0003.
- Git identity is the global `lionel509 <lionelweng@gmail.com>`; never override it.
