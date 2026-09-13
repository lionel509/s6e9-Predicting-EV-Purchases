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
- `v32_hybrid_variants.py MODE` — single-change variants of v27 (`tecv`, `wobble`, `rank`, `linear`, `extra`); all dead, see runs 61–76.
- `v33_hybrid_views.py VIEW [folds] [seed] [lgb|xgb] [init key]` — megayak's feature views C / D (+ our E, F) on our pipeline: centred-window target rates, lift vs the original, coarser income ladders; optional XGBoost learner or an init-score start. View D bags to 0.946215.
- `v34_init_score.py [folds] [seed] [key]` — the hybrid frame boosted from logit of a nested target rate as `init_score`. From `k_inc100` this is the strongest own model: five seeds 0.946395 as a group (run 87), versus 0.946379 for the eight-seed hybrid bag.
- `v35_realmlp.py [folds] [seed] [epochs]` — yekenot's pure-PyTorch RealMLP on Apple MPS (notebook code verbatim, our fold loop). ~2 min per fold; five 10-fold seeds 0.946247 as a group. Re-pull the notebook before regenerating: it changes.
- `postprocess.py <name>` — the four deterministic boundary rules (upper income cliff, income dead zone, commute ≥ 83, 30k zero cell) applied in rank space to a submission; +0.000005 OOF, nested-checked (`pp_rules.py`).
- `nested_check.py` — split-half nested weight fit for two group sets; the check that made blend_v20 the final candidate over v18 after the public board moved the other way.
- `assemble.py <out>` — the blend. Each entry in `GROUPS` is a list of OOF names rank-averaged into one signal plus an optional test-column override; weights by coordinate search over the simplex + Nelder-Mead polish (plain Nelder-Mead stalls past ~10 groups). Writes `submissions/<out>.csv` and `oof_<out>.npy`.
- `stack.py` — nested LR / LightGBM stackers over the group columns; loses to the linear blend.
- `diag_public_oof.py` — AUC, Spearman and nested blend fits of the public OOF files against ours.
- `blend.py` — the older two-to-five-model rank blend; `tune.py` + `tuning.db` — the 09-01 Optuna study (pre-TE frame, stale).
- `run_chain*.sh` — sequential CPU queues (`runs.log`); `run_nn*.sh` — the GPU (MPS) queue for v35, run alongside (`runs_nn.log`). Refits are queued with `until grep -q "chainN done" runs.log; do sleep 30; done; python assemble.py blend_vX`.
- `submissions/` (gitignored) — `oof_<name>.npy` + `<name>.csv` for every run; `pub_*` are the public sources. `leaderboard/` — dated board snapshots.

## Rules of the repo
- Every run gets a row in the note's run log, including the ones that went nowhere.
- Nothing goes to the board without a leak-free OOF above the incumbent blend's. Refits and seed bags have no OOF and ride inside a blend.
- When adding a `GROUPS` entry with a string replace, assert the anchor matched and re-parse the table (`ast`): a comment appended mid-line silently disabled three groups for two refits on 09-13 (run 87).
- Pseudo-labels come only from a model that never saw the validation fold (`v22`, `v28 --pseudo`); un-nested TE and fold-averaged soft labels both inflate OOF by 0.0002–0.0003.
- Git identity is the global `lionel509 <lionelweng@gmail.com>`; never override it.
