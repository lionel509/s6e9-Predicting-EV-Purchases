#!/bin/zsh
# 2026-09-29: does a full-data refit beat the 20-fold average on test? fold 0 of k10 s7 stands in for test (issue #1)
cd "${0:A:h}"
for r in 900 1000 1200; do
  echo "=== HOLDOUT REFIT=$r s7 start $(date +%T)"
  HOLDOUT=1 REFIT=$r .venv/bin/python -u v40_heuljax.py 7 2>&1 | grep --line-buffered -v "^\["
done
echo "=== HOLDOUT N_FOLDS=20 s7 start $(date +%T)"
HOLDOUT=1 N_FOLDS=20 .venv/bin/python -u v40_heuljax.py 7 2>&1 | grep --line-buffered -E "HOLDOUT|OOF|Traceback|Error"
echo "=== refit_ho done $(date +%T)"
