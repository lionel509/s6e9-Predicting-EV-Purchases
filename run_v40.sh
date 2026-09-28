#!/bin/zsh
# v40 heuljax port, three fold seeds back to back (issue #1)
cd "${0:A:h}"
for s in ${@:-7 2026 101}; do
  echo "=== v40_heuljax.py $s start $(date +%T)"
  .venv/bin/python -u v40_heuljax.py $s 2>&1 | grep --line-buffered -v "^\["
  echo "=== exit $(date +%T)"
done
echo "=== v40 done"
