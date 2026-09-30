# CLEANUP

How to tear this repo back down to tracked files only once the competition is over.
Everything below is regenerable: data by the Kaggle CLI, OOFs and submissions by the scripts.

## What this project leaves behind

| Path / thing | Created by | Size note |
|---|---|---|
| `.venv/` | `uv sync` (pyproject.toml, uv.lock) | torch, lightgbm, xgboost, catboost: large (GBs) |
| `__pycache__/`, `src/**/__pycache__/` | any script | small |
| `*.egg-info` | editable install of `s6e9_ev_purchases` | small |
| `data/` (`train.csv`, `test.csv`, `orig/`, `public/`) | `kaggle competitions download`, `kaggle datasets download`, `import_public.py` | competition data, hundreds of MB; not printed or committed |
| `submissions/` (`oof_<name>.npy`, `<name>.csv`, `pub_*`) | every `v*.py`, `assemble.py`, `refit_full.py`, `postprocess.py`, `stack.py` | largest output; one OOF + one CSV per run |
| `*.out` | `run_chain*.sh` / `run_nn*.sh` console captures | small |
| `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/` | tooling, if run | small |

Tracked and NOT touched by cleanup: `runs.log`, `runs_nn.log`, `tune.log`, `tuning.db`, `leaderboard/*.csv`, `uv.lock`.

## Preview

```
git clean -ndX -e '!.env' -e '!.env.*'
```

Lists every ignored file that would be removed, keeping secrets.

## Clean the repo

```
git clean -fdX -e '!.env' -e '!.env.*'
```

`-X` removes only ignored files, so tracked files (including logs and leaderboard snapshots) stay. If `uv` left a `.venv` outside git's view (for example a symlink), also run `rm -rf .venv`.

## Outside the repo

- Kaggle CLI token: `~/.kaggle/access_token` (README setup). Shared across projects; remove by hand only if you are done with Kaggle entirely: `rm ~/.kaggle/access_token`.
- uv's package cache (filled by `uv sync`, shared and optional): `uv cache clean`.
- No project-specific `~/.cache/<name>` or model-weight cache is written: the code trains from scratch (RealMLP and TabM use Apple MPS, no pretrained downloads).

## Secrets

The commands above deliberately keep `.env` and `.env.*`. The Kaggle token lives in `~/.kaggle/`, outside the repo. To really remove a local `.env`: `rm -f .env .env.*`.
