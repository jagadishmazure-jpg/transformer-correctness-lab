import pytest
import torch

from tclab.encoder import TinyEncoder, TinySeq2Seq, pool


def _batch():
    torch.manual_seed(0)
    ids = torch.randint(1, 50, (2, 10))
    pad = torch.zeros(2, 10, dtype=torch.bool)
    pad[0, 6:] = True
    ids2 = ids.clone()
    ids2[0, 6:] = torch.randint(1, 50, (4,))
    return ids, ids2, pad


@pytest.mark.parametrize("bug", [None, "no_pad_mask", "query_mask"])
def test_encoder_pad_tokens_do_not_affect_real_tokens(bug):
    ids, ids2, pad = _batch()
    e = TinyEncoder(bug=bug)
    with torch.no_grad():
        d = (e(ids, pad)[0, :6] - e(ids2, pad)[0, :6]).abs().max()
    assert (d < 1e-5) == (bug is None)


def test_encoder_is_bidirectional():
    ids, _, pad = _batch()
    ids3 = ids.clone()
    ids3[1, 9] = (ids3[1, 9] + 1) % 50
    e = TinyEncoder()
    with torch.no_grad():
        assert (e(ids, pad)[1, 0] - e(ids3, pad)[1, 0]).abs().max() > 1e-4  # encoders SHOULD see the future


def test_padded_equals_unpadded():
    ids, _, pad = _batch()
    e = TinyEncoder()
    with torch.no_grad():
        a = e(ids, pad)[0, :6]
        b = e(ids[:1, :6], torch.zeros(1, 6, dtype=torch.bool))[0]
    assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("how", ["mean", "cls"])
def test_pooling_ignores_padding(how):
    ids, ids2, pad = _batch()
    e = TinyEncoder()
    with torch.no_grad():
        assert torch.allclose(pool(e(ids, pad), pad, how), pool(e(ids2, pad), pad, how), atol=1e-5)


def test_buggy_mean_pooling_detected():
    ids, ids2, pad = _batch()
    e = TinyEncoder()
    with torch.no_grad():
        assert (pool(e(ids, pad), pad, bug=True) - pool(e(ids2, pad), pad, bug=True)).abs().max() > 1e-3


@pytest.mark.parametrize("bug", [None, "no_causal_decoder", "cross_ignores_pad"])
def test_seq2seq_masks(bug):
    ids, ids2, pad = _batch()
    tgt = torch.randint(1, 50, (2, 8))
    tgt2 = tgt.clone()
    tgt2[:, 4:] = torch.randint(1, 50, (2, 4))
    m = TinySeq2Seq(bug=bug)
    with torch.no_grad():
        fut = (m(ids, pad, tgt)[:, :4] - m(ids, pad, tgt2)[:, :4]).abs().max()
        padd = (m(ids, pad, tgt) - m(ids2, pad, tgt)).abs().max()
    assert (fut > 1e-5) == (bug == "no_causal_decoder")
    assert (padd > 1e-5) == (bug == "cross_ignores_pad")
