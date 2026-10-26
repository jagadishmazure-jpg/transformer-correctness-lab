import torch

from tclab.positions import (
    alibi_bias,
    apply_rope,
    learned_position_cliff,
    max_rope_angle,
    position_ids_from_padding,
    rope_score,
    sinusoid_extrapolates,
    sinusoidal,
)


def test_rope_relative_property():
    torch.manual_seed(0)
    q, k = torch.randn(16).double(), torch.randn(16).double()
    s = [rope_score(q, k, i, i - 3) for i in (3, 10, 50, 500)]
    assert max(s) - min(s) < 1e-9


def test_rope_preserves_norm():
    x = torch.randn(1, 8, 16).double()
    assert torch.allclose(apply_rope(x, torch.arange(8)).norm(dim=-1), x.norm(dim=-1))


def test_rope_kv_cache_offset():
    x = torch.randn(1, 6, 16).double()
    full = apply_rope(x, torch.arange(6))
    assert torch.allclose(full[:, 5:], apply_rope(x[:, 5:], torch.tensor([5])))
    assert not torch.allclose(full[:, 5:], apply_rope(x[:, 5:], torch.tensor([0])))  # offset bug


def test_learned_position_cliff():
    r = learned_position_cliff()
    assert r["acc_in_range"] == 1.0 and r["acc_beyond"] < 0.9


def test_sinusoidal_defined_beyond_training_and_offset_consistent():
    assert sinusoid_extrapolates(4096, 16) <= 1.0
    assert torch.allclose(sinusoidal(10, 16)[7:], sinusoidal(3, 16, offset=7), atol=1e-5)


def test_alibi_bias_monotone_and_zero_on_diagonal():
    b = alibi_bias(6, 4)
    assert torch.all(b.diagonal(dim1=1, dim2=2) == 0)
    assert torch.all(b[:, 5, :-1].diff(dim=-1) > 0)
    assert b[0, 5, 0] < b[3, 5, 0]  # head 0 has the steepest slope


def test_interpolation_and_ntk_keep_angles_in_trained_range():
    trained = max_rope_angle(2048, 64)
    assert max_rope_angle(8192, 64) > 3 * trained
    assert abs(max_rope_angle(8192, 64, interp=0.25) - trained) < 1e-3
    assert max_rope_angle(8192, 64, ntk_scale=4.0) <= trained * 1.01


def test_left_padding_position_ids():
    pad = torch.tensor([[True, True, False, False, False], [False] * 5])
    assert position_ids_from_padding(pad).tolist() == [[0, 0, 0, 1, 2], [0, 1, 2, 3, 4]]
    assert position_ids_from_padding(pad, bug=True)[0, 2:].tolist() == [2, 3, 4]


def test_wrong_position_ids_change_model_output():
    from leaktest.model import build
    m = build("correct")
    ids = torch.randint(0, 64, (1, 5))
    with torch.no_grad():
        assert not torch.allclose(m(ids), m(ids, position_ids=torch.arange(2, 7)))
