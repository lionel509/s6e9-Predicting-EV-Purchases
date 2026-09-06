#!/bin/sh
cd "$(dirname "$0")"
for s in "v19_pseudo2.py v9_pseudo_v6b_round 1 naji" "v19_pseudo2.py v9_pseudo_v6b_round 1 slow" "v20_pseudo_xgb.py v9_pseudo_v6b_round 1" "v21_pseudo_cat.py v9_pseudo_v6b_round"; do
  echo "=== $s start $(date -u +%H:%M:%S)" >> runs.log
  .venv/bin/python $s 2>&1 | grep --line-buffered -v -i warning | grep --line-buffered -v eval_set | tee -a runs.log
  echo "=== exit $(date -u +%H:%M:%S)" >> runs.log
done
echo "=== chain4 done" >> runs.log
