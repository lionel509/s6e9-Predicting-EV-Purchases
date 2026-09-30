#!/bin/zsh
# 2026-09-30 (issue #1): heuljax XGB + GPT-2 token MSTE keys (XTOK=1), k20; s7 first vs plain k20 s7 0.946333, then more seeds
cd "${0:A:h}/.."
for s in ${@:-7}; do
  echo "=== xtok k20 s$s start $(date +%T)"
  XTOK=1 N_FOLDS=20 nice -n 10 .venv/bin/python -u models/v40_heuljax.py $s 2>&1 | grep --line-buffered -E "OOF 0\.|Traceback|Error"
  echo "=== exit $(date +%T)"
done
echo "=== chain41 done $(date +%T)"
