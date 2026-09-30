#!/bin/sh
cd "$(dirname "$0")"
while pgrep -f "run_nn[1234].sh" > /dev/null; do sleep 20; done
echo "=== models/v36_tabm.py 5 42 start $(date -u +%H:%M:%S)" >> logs/runs_nn.log
.venv/bin/python models/v36_tabm.py 5 42 2>&1 | grep --line-buffered -v -i -E "warning|epoch|best score" | tee -a logs/runs_nn.log
echo "=== exit $(date -u +%H:%M:%S)" >> logs/runs_nn.log
if .venv/bin/python -c "import json,sys; d=json.load(open('submissions/v36_tabm_k5_s42.json')); sys.exit(0 if d['oof_auc'] >= 0.9459 else 1)"; then
  echo "=== models/v36_tabm.py 10 42 start $(date -u +%H:%M:%S)" >> logs/runs_nn.log
  .venv/bin/python models/v36_tabm.py 10 42 2>&1 | grep --line-buffered -v -i -E "warning|epoch|best score" | tee -a logs/runs_nn.log
  echo "=== exit $(date -u +%H:%M:%S)" >> logs/runs_nn.log
else
  echo "=== v36_tabm_k5_s42 oof_auc below 0.9459 gate, skipping 10-fold $(date -u +%H:%M:%S)" >> logs/runs_nn.log
fi
echo "=== models/v35_realmlp.py 20 42 start $(date -u +%H:%M:%S)" >> logs/runs_nn.log
.venv/bin/python models/v35_realmlp.py 20 42 2>&1 | grep --line-buffered -v -i -E "warning|epoch|best score" | tee -a logs/runs_nn.log
echo "=== exit $(date -u +%H:%M:%S)" >> logs/runs_nn.log
echo "=== nn5 done" >> logs/runs_nn.log
