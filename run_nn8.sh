#!/bin/sh
# 2026-09-29: RealMLP at 20 folds on the MPS GPU, alongside the CPU-bound heuljax k20 queue (issue #1)
cd "$(dirname "$0")"
for s in ${@:-42 7 2026}; do
  echo "=== v35_realmlp.py 20 $s 2 start $(date +%T)"
  .venv/bin/python -u v35_realmlp.py 20 $s 2 2>&1 | grep --line-buffered -v -i -E "warning|epoch|best score"
  echo "=== exit $(date +%T)"
done
echo "=== nn8 done"
