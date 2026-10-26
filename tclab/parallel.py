"""Module 3: parallel attention equivalences, simulated on CPU."""

from __future__ import annotations

import torch

from tclab.attention import attention


def blockwise_attention(q, k, v, chunk: int, causal: bool, bug: str | None = None):
    """Ring / blockwise attention with online softmax (the FlashAttention / ring-attention idea).

    Query chunks iterate over key chunks, keeping a running max ``m``, normaliser ``l`` and
    accumulator ``acc``. ``bug``:
    * "no_rescale"        - forgets to rescale acc/l when the running max changes
    * "local_causal"      - applies the causal mask using chunk-local indices (wrong across chunks)
    """
    T, d = q.shape[-2], q.shape[-1]
    out = torch.empty_like(q)
    for qs in range(0, T, chunk):
        qi = q[..., qs : qs + chunk, :]
        m = torch.full(qi.shape[:-1] + (1,), float("-inf"))
        l = torch.zeros_like(m)
        acc = torch.zeros_like(qi)
        for ks in range(0, T, chunk):  # in a ring, this chunk would arrive from the next device
            if causal and ks > qs + chunk - 1:
                break
            kj, vj = k[..., ks : ks + chunk, :], v[..., ks : ks + chunk, :]
            s = qi @ kj.transpose(-2, -1) / d**0.5
            if causal:
                qpos = torch.arange(qi.shape[-2])[:, None] + (0 if bug == "local_causal" else qs)
                kpos = torch.arange(kj.shape[-2])[None, :] + (0 if bug == "local_causal" else ks)
                s = s.masked_fill(kpos > qpos, float("-inf"))
            m_new = torch.maximum(m, s.amax(-1, keepdim=True))
            p = torch.exp(s - m_new).nan_to_num(0.0)
            scale = torch.exp(m - m_new).nan_to_num(0.0)
            if bug == "no_rescale":
                scale = torch.ones_like(scale)
            l = l * scale + p.sum(-1, keepdim=True)
            acc = acc * scale + p @ vj
            m = m_new
        out[..., qs : qs + chunk, :] = acc / l
    return out


def full_attention(q, k, v, causal: bool):
    T = q.shape[-2]
    return attention(q, k, v, torch.ones(T, T, dtype=torch.bool).tril() if causal else None)


def tensor_parallel_heads(x, Wq, Wk, Wv, Wo, n_heads: int, shards: int):
    """Megatron-style: split heads across ``shards`` simulated devices; column-parallel QKV,
    row-parallel output projection, then an all-reduce (sum) of partial outputs."""
    T, D = x.shape
    hd = D // n_heads
    per = n_heads // shards
    partials = []
    for s in range(shards):
        cols = slice(s * per * hd, (s + 1) * per * hd)
        q = (x @ Wq[:, cols]).view(T, per, hd).transpose(0, 1)
        k = (x @ Wk[:, cols]).view(T, per, hd).transpose(0, 1)
        v = (x @ Wv[:, cols]).view(T, per, hd).transpose(0, 1)
        o = full_attention(q, k, v, causal=True).transpose(0, 1).reshape(T, -1)
        partials.append(o @ Wo[cols, :])
    return torch.stack(partials).sum(0)  # all-reduce


def single_device_heads(x, Wq, Wk, Wv, Wo, n_heads: int):
    return tensor_parallel_heads(x, Wq, Wk, Wv, Wo, n_heads, shards=1)


def sequential_rnn_steps(T: int) -> dict:
    """Work vs span: attention does O(T^2) work in O(1) sequential steps per layer (all
    positions in parallel in training); an RNN needs T sequential steps."""
    return {"attention_work": T * T, "attention_span": 1, "rnn_work": T, "rnn_span": T}
