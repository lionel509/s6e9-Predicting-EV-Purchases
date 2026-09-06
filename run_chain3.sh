#!/bin/sh
cd "$(dirname "$0")"
while ! grep -q "chain2 done" runs.log; do sleep 10; done
echo "=== v17_params start $(date -u +%H:%M:%S)" >> runs.log
.venv/bin/python v17_params.py slow leaves32 naji 2>&1 | grep -v -i warning | grep -v eval_set | tee -a runs.log
echo "=== chain3 done" >> runs.log
