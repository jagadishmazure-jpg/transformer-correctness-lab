# Implementation guide: adopt these checks in your own model's CI

## 1. Wrap your model as a function
```python
import torch
from leaktest.checks import perturbation_leak_check

model = load_my_model().eval()          # your model
fn = lambda ids: model(ids).logits      # (B, T) -> (B, T, V)
tokens = torch.randint(0, vocab_size, (2, 32))
report = perturbation_leak_check(fn, tokens, vocab_size, atol=1e-4)
assert not report.leaked, report.summary()
```
Use fp32 and `eval()` (dropout off). Pick `atol` above your float noise: about 1e-5 for fp32 small models, larger for big or reduced-precision models.

## 2. Add the gradient check if you can reach the embeddings
Feed `inputs_embeds` with `requires_grad=True`, backprop `logits[:, t].sum()`, and assert the gradient on positions > t is zero. See `gradient_leak_check`.

## 3. Add KV-cache consistency
Generate logits token by token with your cache and compare with one full forward pass. Run it with left padding too.

## 4. Cover your data path
- Padding: change pad ids, require real-token logits unchanged; padded == unpadded.
- Packing: change one document, require the others unchanged.
- Position ids: assert left-padded rows start at 0.

## 5. Keep it cheap
Use a 1-2 layer config of your architecture with random weights (see [tiny-transformer.md](tiny-transformer.md)); the bug is scale-independent. Run full-size checks nightly if you want.

## 6. Gate on exit codes
```yaml
- run: python -m pytest tests/test_causal_contract.py
```
Fail the build on any leak, NaN or equivalence break. Keep known failures visible as tests marked `xfail` with a reason, rather than deleting them.

## 7. Precision
If you train in fp16/bf16, prefer fused `is_causal` kernels; otherwise use `-inf` masks with softmax upcast to fp32 and handle fully-masked rows explicitly.
