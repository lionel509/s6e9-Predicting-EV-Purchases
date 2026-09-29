#!/bin/zsh
# 2026-09-29 resume step 2 (issue #1): agg + GPT-2 token keys at 20 folds, 3 seeds -> group agg_init100_tok_k20. Starts after chain36.
cd "${0:A:h}"
while kill -0 59521 2>/dev/null; do sleep 30; done
for s in 7 2026 42; do
  echo "=== agg tok k20 s$s start $(date +%T)"
  TOKENS=1 nice -n 10 .venv/bin/python -u v37_hybrid_modes.py agg 20 $s k_inc100 2>&1 | grep --line-buffered -E "OOF|Traceback|Error"
  echo "=== exit $(date +%T)"
done
echo "=== chain37 done"
