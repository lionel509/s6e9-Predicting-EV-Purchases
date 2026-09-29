#!/bin/zsh
# 2026-09-29: GPT-2-token LR (v42 port of heuljax's generator-aware ridge LR), own seeds at 20 and 30 folds (issue #1)
cd "${0:A:h}"
for s in 7 2026 101 202; do
  for k in 20 30; do
    echo "=== v42 N_FOLDS=$k s$s start $(date +%T)"
    N_FOLDS=$k .venv/bin/python -u v42_hjlr.py $s 2>&1 | grep --line-buffered -E "OOF 0\.|Traceback|Error"
  done
done
echo "=== v42 done $(date +%T)"
