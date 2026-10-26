#!/usr/bin/env bash
# Fails if README's test count or any captured output differs from a fresh run.
set -euo pipefail
n=$(python scripts/count_tests.py)
echo "collected tests: $n"
grep -q "\*\*$n tests\*\*" README.md || { echo "README test count is stale (expected **$n tests**)"; exit 1; }
tmp=$(mktemp -d)
bash scripts/regen_outputs.sh "$tmp"
# fp rounding of tiny diffs (~1e-7) can vary across CPUs; compare everything else exactly
norm() { sed -E 's/[0-9]\.[0-9]{3}e-0[5-9]/~0/g' "$1"; }
fail=0
for f in docs/output/*.txt; do
  diff <(norm "$f") <(norm "$tmp/$(basename "$f")") >/dev/null || { echo "stale: $f"; diff <(norm "$f") <(norm "$tmp/$(basename "$f")") | head; fail=1; }
done
python scripts/embed_outputs.py --check || fail=1
[ $fail -eq 0 ] && echo "docs outputs match a fresh run"
exit $fail
