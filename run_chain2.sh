#!/bin/sh
# Waits for the v14 ablation chain, then runs v15 (TE smoothing) and v16 (residual TE) sequentially into runs.log
cd "$(dirname "$0")"
while pgrep -f "v14_ablate.py" >/dev/null; do sleep 5; done
for s in "v15_te_m.py 5 50 dual" "v16_resid_te.py"; do
  n=$(echo $s | cut -d' ' -f1 | sed 's/.py//'); echo "=== $n start $(date -u +%H:%M:%S)" >> runs.log
  .venv/bin/python $s 2>&1 | grep -v -i warning | grep -v eval_set | tee -a runs.log
  echo "=== $n exit $(date -u +%H:%M:%S)" >> runs.log
done
echo "=== chain2 done" >> runs.log
