#!/bin/sh
cd "$(dirname "$0")"
for s in "v27_hybrid.py 5 42"; do
  echo "=== $s start $(date -u +%H:%M:%S)" >> runs.log
  .venv/bin/python $s 2>&1 | grep --line-buffered -v -i warning | tee -a runs.log
  echo "=== exit $(date -u +%H:%M:%S)" >> runs.log
done
echo "=== chain6 done" >> runs.log
