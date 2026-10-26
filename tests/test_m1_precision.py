import torch

from leaktest.checks import perturbation_leak_check
from leaktest.model import build
from leaktest.precision import finite_mask_too_small_demo, fp16_min_overflow_demo, fully_masked_row_demo


def test_neg_inf_fully_masked_row_gives_nan():
    assert fully_masked_row_demo(torch.float16)["neg_inf_has_nan"]


def test_finite_mask_fully_masked_row_leaks_uniformly():
    r = fully_masked_row_demo(torch.float16)
    assert r["finite_leaks_to_future"] and r["finite_row0"] == [0.25] * 4


def test_bf16_has_same_fully_masked_row_problem():
    assert fully_masked_row_demo(torch.bfloat16)["neg_inf_has_nan"]


def test_small_additive_mask_leaks_but_fp32_neg_inf_does_not():
    r = finite_mask_too_small_demo()
    assert r["mask_-20_future_weight"] > 0.99 and r["neg_inf_fp32_future_weight"] == 0.0


def test_fp16_min_overflows_to_neg_inf():
    assert fp16_min_overflow_demo()["is_inf"]


def test_fused_causal_in_bf16_has_no_leak():
    m = build("fused").to(torch.bfloat16)
    r = perturbation_leak_check(m, torch.randint(0, 64, (2, 10)), 64, atol=0.0)
    assert not r.leaked
