# Module 2: Encoder and encoder-decoder masks

Encoders (BERT-style) are **bidirectional**: every token may see every *real* token, including later ones. The contract is about **padding**: pad tokens must not affect real-token outputs.

| Check | Correct | Injected bug |
|---|---|---|
| Change pad-token ids, real outputs fixed | key-padding mask `(~pad)[:, None, :]` | `no_pad_mask`; `query_mask` (masks pad *rows* instead of pad *keys*, a common confusion) |
| Padded row == same sequence unpadded | yes | |
| Encoder really is bidirectional | changing a later token changes token 0 | (sanity check that we did not over-mask) |
| Mean pooling ignores pads | masked sum / real count | naive `h.mean(1)` averages pads |
| [CLS] pooling | position 0 unaffected by pads | |
| Seq2seq decoder self-attention | causal | `no_causal_decoder` sees future targets |
| Seq2seq cross-attention | respects encoder padding | `cross_ignores_pad` |

## Real output: `python -m tclab encoder`
<!-- OUTPUT:encoder.txt -->
```text
[encoder] change pad-token ids, real-token outputs must not move
  OK  bug=None         max_diff_real_tokens: 0.000e+00
  OK  bug=no_pad_mask  max_diff_real_tokens: 5.962e-01
  OK  bug=query_mask   max_diff_real_tokens: 5.933e-01
  OK  masked mean pooling diff: 0.000e+00
  OK  buggy mean pooling (averages pads) diff: 6.938e-01
[seq2seq] perturb future target tokens / source pads
  OK  bug=None               future_diff=0.000e+00 src_pad_diff: 0.000e+00
  OK  bug=no_causal_decoder  future_diff=6.968e-01 src_pad_diff: 0.000e+00
  OK  bug=cross_ignores_pad  future_diff=0.000e+00 src_pad_diff: 4.811e-01

RESULT: encoder=pass
```
<!-- /OUTPUT -->

Code: `tclab/encoder.py`. Tests: `tests/test_m2_encoder.py`.
