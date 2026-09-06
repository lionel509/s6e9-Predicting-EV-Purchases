#!/bin/sh
cd "$(dirname "$0")"
for s in "v22_pseudo_clean.py naji 1 7 0" "v22_pseudo_clean.py naji 1 2026 0" "v26_cat_refit.py 1330 42 101 202"; do
  echo "=== $s start $(date -u +%H:%M:%S)" >> runs.log
  .venv/bin/python $s 2>&1 | grep --line-buffered -v -i warning | tee -a runs.log
  echo "=== exit $(date -u +%H:%M:%S)" >> runs.log
done
echo "=== chain5 done" >> runs.log
