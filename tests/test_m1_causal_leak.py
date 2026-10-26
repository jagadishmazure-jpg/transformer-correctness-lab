"""Module 1: causal mask leak - correct masks pass, every injected bug is caught."""

import pytest
import torch

from leaktest.__main__ import main as cli_main
from leaktest.checks import gradient_leak_check, kv_cache_consistency, perturbation_leak_check
from leaktest.masks import packed_causal, padded_causal
from leaktest.model import BUGGY_MODES, CORRECT_MODES, build

SDPA_MODES = [m for m in CORRECT_MODES + BUGGY_MODES if "mha" not in m]


@pytest.mark.parametrize("mode", CORRECT_MODES)
def test_correct_masks_pass_perturbation(mode, tokens):
    m = build(mode)
    r = perturbation_leak_check(m, tokens, m.cfg.vocab_size)
    assert not r.leaked, r.summary()


@pytest.mark.parametrize("mode", BUGGY_MODES)
def test_buggy_masks_detected_by_perturbation(mode, tokens):
    m = build(mode)
    r = perturbation_leak_check(m, tokens, m.cfg.vocab_size)
    assert r.leaked and r.first_leak_position == 0


@pytest.mark.parametrize("mode", SDPA_MODES)
def test_gradient_check_agrees(mode, tokens):
    r = gradient_leak_check(build(mode), tokens)
    assert r.leaked == (mode in BUGGY_MODES)


@pytest.mark.parametrize("mode", SDPA_MODES)
def test_kv_cache_matches_full_forward_only_when_causal(mode, tokens):
    r = kv_cache_consistency(build(mode), tokens)
    assert r.leaked == (mode in BUGGY_MODES)


def test_off_by_one_reports_position():
    m = build("buggy_off_by_one")
    r = perturbation_leak_check(m, torch.randint(0, 64, (1, 8)), 64)
    assert r.first_leak_position == 0 and "position <= 0" in r.details[0]


@pytest.mark.parametrize("side", ["right", "left"])
def test_padding_plus_causal(side):
    """Changing pad-token ids must not change real-token logits; batched == unbatched."""
    m = build("correct")
    torch.manual_seed(1)
    real = torch.randint(1, 64, (1, 6))
    T = 10
    pad = torch.zeros(2, T, dtype=torch.bool)
    ids = torch.randint(1, 64, (2, T))
    sl = slice(6, T) if side == "right" else slice(0, 4)
    rl = slice(0, 6) if side == "right" else slice(4, T)
    pad[0, sl] = True
    ids[0, rl] = real
    pos = torch.arange(T).expand(2, T).clone()
    if side == "left":
        pos[0] = (torch.arange(T) - 4).clamp_min(0)
    mask = padded_causal(pad)
    ids2 = ids.clone()
    ids2[0, sl] = torch.randint(1, 64, (T - 6,))
    with torch.no_grad():
        a = m(ids, mask, pos)[0, rl]
        b = m(ids2, mask, pos)[0, rl]
        solo = m(real)[0]
    assert not torch.isnan(m(ids, mask, pos)).any()
    assert torch.allclose(a, b, atol=1e-5)
    assert torch.allclose(a, solo, atol=1e-5)  # variable-length batch == unbatched


def test_padding_without_key_mask_leaks():
    m = build("correct")
    ids = torch.randint(1, 64, (1, 8))
    ids2 = ids.clone()
    ids2[0, :3] = (ids2[0, :3] + 1) % 64  # left pads changed
    with torch.no_grad():  # plain causal mask, pads NOT hidden -> real tokens read pads
        d = (m(ids)[0, 3:] - m(ids2)[0, 3:]).abs().max()
    assert d > 1e-3


def _packed(mask_fn):
    m = build("correct")
    doc = torch.tensor([[0] * 5 + [1] * 5])
    ids = torch.randint(0, 64, (1, 10))
    ids2 = ids.clone()
    ids2[0, :5] = (ids2[0, :5] + 7) % 64  # change document A only
    pos = torch.cat([torch.arange(5), torch.arange(5)])[None]
    with torch.no_grad():
        a = m(ids, mask_fn(doc), pos)[0, 5:]
        b = m(ids2, mask_fn(doc), pos)[0, 5:]
    return (a - b).abs().max().item()


def test_packed_sequences_block_diagonal_no_cross_doc_leak():
    assert _packed(packed_causal) < 1e-6


def test_packed_sequences_plain_causal_leaks_across_documents():
    plain = lambda doc: torch.ones(10, 10, dtype=torch.bool).tril()[None, None]
    assert _packed(plain) > 1e-3


def test_cli_all_exit_zero(capsys):
    assert cli_main(["--mask", "all"]) == 0
    assert "9/9 modes classified as expected" in capsys.readouterr().out


def test_cli_single_buggy_exit_one():
    assert cli_main(["--mask", "buggy_transpose"]) == 1
