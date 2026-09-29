#!/bin/zsh
# v41: heuljax with self-distillation (DISTILL=1, 10 folds) and 20 folds, alternating seeds (issue #1)
cd "${0:A:h}"
seeds=(7 2026 101 202); (( $# )) && seeds=($@)
for s in $seeds; do
  for cfg in "DISTILL=1" "N_FOLDS=20"; do
    echo "=== $cfg v40_heuljax.py $s start $(date +%T)"
    env $cfg .venv/bin/python -u v40_heuljax.py $s 2>&1 | grep --line-buffered -v "^\["
    echo "=== exit $(date +%T)"
  done
done
echo "=== v41 done"
