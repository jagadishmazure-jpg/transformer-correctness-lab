import pytest
import torch

from leaktest.model import build
from tclab.parallel import blockwise_attention, full_attention, sequential_rnn_steps, single_device_heads, tensor_parallel_heads


@pytest.fixture
def qkv():
    torch.manual_seed(0)
    return [torch.randn(2, 20, 16) * 3 for _ in range(3)]


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("chunk", [1, 4, 6, 20])
def test_blockwise_equals_full(qkv, causal, chunk):
    assert torch.allclose(blockwise_attention(*qkv, chunk, causal), full_attention(*qkv, causal), atol=1e-4)


@pytest.mark.parametrize("causal", [False, True])
def test_missing_online_softmax_rescale_detected(qkv, causal):
    assert not torch.allclose(blockwise_attention(*qkv, 6, causal, "no_rescale"), full_attention(*qkv, causal), atol=1e-3)


def test_chunk_local_causal_indices_detected(qkv):
    out = blockwise_attention(*qkv, 6, True, "local_causal")
    assert not torch.allclose(out, full_attention(*qkv, True), atol=1e-3)


def test_blockwise_causal_has_no_future_leak(qkv):
    q, k, v = qkv
    k2, v2 = k.clone(), v.clone()
    k2[:, 10:] += 5
    v2[:, 10:] += 5
    a = blockwise_attention(q, k, v, 6, True)[:, :10]
    b = blockwise_attention(q, k2, v2, 6, True)[:, :10]
    assert torch.allclose(a, b)


@pytest.mark.parametrize("shards", [2, 4, 8])
def test_tensor_parallel_heads_equivalent(shards):
    torch.manual_seed(0)
    x = torch.randn(10, 32)
    W = [torch.randn(32, 32) * 0.2 for _ in range(4)]
    assert torch.allclose(tensor_parallel_heads(x, *W, 8, shards), single_device_heads(x, *W, 8), atol=1e-4)


def test_batch_invariance():
    m = build("correct")
    torch.manual_seed(0)
    a = torch.randint(0, 64, (1, 12))
    with torch.no_grad():
        solo = m(a)
        for n in (2, 8):
            assert torch.allclose(solo, m(torch.cat([a, torch.randint(0, 64, (n - 1, 12))]))[:1], atol=1e-5)


def test_work_vs_span():
    r = sequential_rnn_steps(128)
    assert r["attention_span"] == 1 and r["rnn_span"] == 128 and r["attention_work"] == 128 * 128
