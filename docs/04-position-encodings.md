# Module 4: Position encodings

| Encoding | Property tested | Failure mode shown |
|---|---|---|
| Sinusoidal | defined for any position, bounded by 1; offset-consistent (`sinusoidal(3, offset=7)` == rows 7-9) | |
| Learned absolute | | **position cliff**: positions past the trained length were never trained. On a toy parity task: 100% accuracy inside the trained 16 positions, 31% beyond |
| RoPE | score depends only on i - j (spread ~1e-15 across 4 pairs with i-j = 3); rotation keeps vector norm | **KV-cache offset bug**: rotating the new token at position 0 instead of its real position |
| ALiBi | zero on the diagonal, penalty grows with distance, steeper slope for earlier heads | |
| Position interpolation / NTK scaling | at 4x the trained length, the lowest-frequency RoPE angle goes from 0.3 to 1.1 rad (never seen in training); interpolation x0.25 or NTK base scaling x4 bring it back to 0.3 | |
| Position ids with left padding | real tokens start at 0 | `arange` ids shift real tokens by the pad count; a test shows wrong ids change model output |

## Real output: `python -m tclab positions`
<!-- OUTPUT:positions.txt -->
```text
[positions]
  OK  RoPE score depends only on i-j (pairs with i-j=3) spread: 1.776e-15
  OK  RoPE decode step with correct offset matches full: True
  OK  RoPE decode step with offset=0 (bug) differs: True
  OK  learned absolute position cliff: {'acc_in_range': 1.0, 'acc_beyond': 0.3125}
  OK  ALiBi penalty grows with distance: [-0.25, -0.1875, -0.125, -0.0625, 0.0]
  info max RoPE angle at T=8192 d=64 [plain]: 1.1 rad (trained T=2048: 0.3)
  info max RoPE angle at T=8192 d=64 [interp x0.25]: 0.3 rad (trained T=2048: 0.3)
  info max RoPE angle at T=8192 d=64 [ntk x4]: 0.3 rad (trained T=2048: 0.3)
  OK  left-pad position ids (real tokens start at 0): [[0, 0, 0, 1, 2]]
  OK  buggy arange position ids: [[0, 1, 2, 3, 4]]

RESULT: positions=pass
```
<!-- /OUTPUT -->

Code: `tclab/positions.py`. Tests: `tests/test_m4_positions.py`.
