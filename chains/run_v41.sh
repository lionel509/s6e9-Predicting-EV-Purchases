#!/bin/zsh
# v41: heuljax at 20 folds (self-distillation was tried and dropped, see issue #1)
cd "${0:A:h}/.."
seeds=(7 2026 101 202); (( $# )) && seeds=($@)
for s in $seeds; do
  for cfg in "N_FOLDS=20"; do   # DISTILL=1 dropped 2026-09-28: lost 6/6 folds on s7
    echo "=== $cfg models/v40_heuljax.py $s start $(date +%T)"
    env $cfg .venv/bin/python -u models/v40_heuljax.py $s 2>&1 | grep --line-buffered -v "^\["
    echo "=== exit $(date +%T)"
  done
done
echo "=== v41 done"
