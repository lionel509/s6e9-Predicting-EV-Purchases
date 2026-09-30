#!/bin/zsh
# 2026-09-29 night (issue #1): EXTRA key sets on the token init100 model, k10 s7, vs init100_tok_k10_s7 0.946446
cd "${0:A:h}"
X=$1
echo "=== EXTRA=$X start $(date +%T)"
TOKENS=1 EXTRA=$X nice -n 10 .venv/bin/python -u v34_init_score.py 10 7 k_inc100 2>&1 | grep --line-buffered -E "OOF|Traceback|Error|fold [0-9]+:"
echo "=== EXTRA=$X done $(date +%T)"
