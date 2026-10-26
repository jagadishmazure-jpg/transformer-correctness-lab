#!/usr/bin/env bash
# Regenerate every captured output under docs/output from real runs.
set -euo pipefail
export PYTHONWARNINGS=ignore
out=${1:-docs/output}
mkdir -p "$out"
for m in leak encoder parallel positions lora; do python -m tclab "$m" > "$out/$m.txt"; done
python -m leaktest --mask buggy_transpose > "$out/leak_buggy_transpose.txt" || true
python -m leaktest --mask correct > "$out/leak_correct.txt"
python -m leaktest --mask buggy_inverted_bool > "$out/leak_inverted.txt" || true
python - > "$out/precision.txt" <<'PY'
import torch
from leaktest.precision import finite_mask_too_small_demo, fp16_min_overflow_demo, fully_masked_row_demo
print('fully_masked_row fp16:', fully_masked_row_demo(torch.float16))
print('fully_masked_row bf16:', fully_masked_row_demo(torch.bfloat16))
print('finite_mask_too_small fp16:', finite_mask_too_small_demo())
print('fp16_min_overflow:', fp16_min_overflow_demo())
PY
