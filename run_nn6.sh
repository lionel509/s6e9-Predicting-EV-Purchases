#!/bin/sh
# 2026-09-27 Fable end-game list: idea 2 (RealMLP from the init100 margin), then idea 6 (TabM seeds, gated)
cd "$(dirname "$0")"
until grep -q "nn5 done" runs_nn.log; do sleep 30; done
run() { echo "=== $* start $(date -u +%H:%M:%S)" >> runs_nn.log; .venv/bin/python "$@" 2>&1 | grep --line-buffered -v -i -E "warning|epoch|best score" | tee -a runs_nn.log; echo "=== exit $(date -u +%H:%M:%S)" >> runs_nn.log; }
gate() { .venv/bin/python -c "import json,sys; sys.exit(0 if json.load(open('submissions/$1.json'))['oof_auc'] >= 0.94615 else 1)" 2>/dev/null; }
run v35_realmlp.py 10 42 2 k_inc100
if gate v35_realmlp_k10_s42_initinc100; then run v35_realmlp.py 10 7 2 k_inc100; run v35_realmlp.py 10 2026 2 k_inc100; else echo "=== realmlp_init100 below gate" >> runs_nn.log; fi
if gate v36_tabm_k10_s42; then run v36_tabm.py 10 7; run v36_tabm.py 10 2026; else echo "=== tabm below gate / missing" >> runs_nn.log; fi
echo "=== nn6 done" >> runs_nn.log
