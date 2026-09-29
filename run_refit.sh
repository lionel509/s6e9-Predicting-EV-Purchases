#!/bin/zsh
# 2026-09-29: full-train heuljax refits for the hj_k20 test column (50/50 with the 20-fold average; holdout run_refit_ho.sh, issue #1)
cd "${0:A:h}"
seeds=(7 2026 101 202 303 404 505 42); (( $# )) && seeds=($@)
for s in $seeds; do
  echo "=== REFIT=1500 s$s start $(date +%T)"
  REFIT=1500 .venv/bin/python -u v40_heuljax.py $s 2>&1 | grep --line-buffered -E "written|Traceback|Error"
done
echo "=== refit done $(date +%T)"
