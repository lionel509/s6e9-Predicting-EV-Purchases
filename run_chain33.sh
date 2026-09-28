#!/bin/sh
# 2026-09-27: v39 stacked features (heuljax's inner-fold basic-model predictions; Fable spec, MiMo Pro build)
cd "$(dirname "$0")"
for s in "v39_stacked_features.py tn 10 42" "v39_stacked_features.py tn_initlr 10 42"; do
  echo "=== $s start $(date -u +%H:%M:%S)" >> runs.log
  .venv/bin/python $s 2>&1 | grep --line-buffered -v -i -E "warning|eval_set|converge|increase the number" | tee -a runs.log
  echo "=== exit $(date -u +%H:%M:%S)" >> runs.log
done
echo "=== chain33 done" >> runs.log
