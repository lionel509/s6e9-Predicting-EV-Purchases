#!/bin/sh
# Sequential so the runs do not fight over the 12 cores. Output: logs/runs.log
cd "$(dirname "$0")"
for s in v6_te_variants v7_xgb v8_catboost; do
  echo "=== $s start $(date -u +%H:%M:%S)"
  .venv/bin/python $s.py 2>&1 | grep -v -i warning | grep -v eval_set
  echo "=== $s exit $(date -u +%H:%M:%S)"
done
echo "=== chain done"
