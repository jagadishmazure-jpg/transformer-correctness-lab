"""python -m tclab {leak,encoder,parallel,positions,lora,all}"""

from __future__ import annotations

import argparse
import sys

import torch


def _line(ok_expected: bool, name: str, value) -> bool:
    print(f"  {'OK ' if ok_expected else 'BAD'} {name}: {value}")
    return ok_expected


def cmd_leak(_a) -> bool:
    from leaktest.__main__ import main

    return main(["--mask", "all"]) == 0


def cmd_encoder(_a) -> bool:
    from tclab.encoder import TinyEncoder, TinySeq2Seq, pool

    torch.manual_seed(0)
    ids = torch.randint(1, 50, (2, 10))
    pad = torch.zeros(2, 10, dtype=torch.bool)
    pad[0, 6:] = True
    ids2 = ids.clone()
    ids2[0, 6:] = torch.randint(1, 50, (4,))
    ok = True
    print("[encoder] change pad-token ids, real-token outputs must not move")
    for bug in (None, "no_pad_mask", "query_mask"):
        e = TinyEncoder(bug=bug)
        with torch.no_grad():
            d = (e(ids, pad)[0, :6] - e(ids2, pad)[0, :6]).abs().max().item()
        ok &= _line((d < 1e-5) == (bug is None), f"bug={bug!s:12} max_diff_real_tokens", f"{d:.3e}")
    e = TinyEncoder()
    with torch.no_grad():
        h = e(ids, pad)
        good = (pool(h, pad) - pool(e(ids2, pad), pad)).abs().max().item()
        bad = (pool(h, pad, bug=True) - pool(e(ids2, pad), pad, bug=True)).abs().max().item()
    ok &= _line(good < 1e-5, "masked mean pooling diff", f"{good:.3e}")
    ok &= _line(bad > 1e-3, "buggy mean pooling (averages pads) diff", f"{bad:.3e}")
    print("[seq2seq] perturb future target tokens / source pads")
    tgt = torch.randint(1, 50, (2, 8))
    tgt2 = tgt.clone()
    tgt2[:, 4:] = torch.randint(1, 50, (2, 4))
    for bug in (None, "no_causal_decoder", "cross_ignores_pad"):
        m = TinySeq2Seq(bug=bug)
        with torch.no_grad():
            fut = (m(ids, pad, tgt)[:, :4] - m(ids, pad, tgt2)[:, :4]).abs().max().item()
            padd = (m(ids, pad, tgt) - m(ids2, pad, tgt)).abs().max().item()
        clean = fut < 1e-5 and padd < 1e-5
        ok &= _line(clean == (bug is None), f"bug={bug!s:18} future_diff={fut:.3e} src_pad_diff", f"{padd:.3e}")
    return ok


def cmd_parallel(_a) -> bool:
    from tclab.parallel import (
        blockwise_attention,
        full_attention,
        sequential_rnn_steps,
        single_device_heads,
        tensor_parallel_heads,
    )

    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 20, 16) * 3 for _ in range(3))
    ok = True
    print("[parallel] blockwise/ring attention (chunk=6) vs full attention")
    for causal in (False, True):
        ref = full_attention(q, k, v, causal)
        for bug in (None, "no_rescale", "local_causal"):
            if bug == "local_causal" and not causal:
                continue
            d = (blockwise_attention(q, k, v, 6, causal, bug) - ref).abs().max().item()
            ok &= _line((d < 1e-4) == (bug is None), f"causal={causal!s:5} bug={bug!s:12} max_diff", f"{d:.3e}")
    x = torch.randn(10, 32)
    W = [torch.randn(32, 32) * 0.2 for _ in range(4)]
    for shards in (2, 4):
        d = (tensor_parallel_heads(x, *W, n_heads=8, shards=shards) - single_device_heads(x, *W, n_heads=8)).abs().max().item()
        ok &= _line(d < 1e-4, f"tensor-parallel heads shards={shards} max_diff", f"{d:.3e}")
    from leaktest.model import build

    m = build("correct")
    a = torch.randint(0, 64, (1, 12))
    with torch.no_grad():
        solo = m(a)
        batched = m(torch.cat([a, torch.randint(0, 64, (7, 12))]))[:1]
    d = (solo - batched).abs().max().item()
    ok &= _line(d < 1e-5, "batch invariance (1 vs 8 samples) max_diff", f"{d:.3e}")
    print(f"  info work/span for T=1024: {sequential_rnn_steps(1024)}")
    return ok


def cmd_positions(_a) -> bool:
    from tclab.positions import (
        alibi_bias,
        apply_rope,
        learned_position_cliff,
        max_rope_angle,
        position_ids_from_padding,
        rope_score,
    )

    torch.manual_seed(0)
    ok = True
    q, k = torch.randn(16).double(), torch.randn(16).double()
    s = [rope_score(q, k, i, j) for i, j in ((5, 2), (13, 10), (40, 37))]
    spread = max(s) - min(s)
    print("[positions]")
    ok &= _line(spread < 1e-9, "RoPE score depends only on i-j (pairs with i-j=3) spread", f"{spread:.3e}")
    x = torch.randn(1, 6, 16).double()
    full = apply_rope(x, torch.arange(6))
    cache_ok = apply_rope(x[:, 5:], torch.tensor([5]))
    cache_bug = apply_rope(x[:, 5:], torch.tensor([0]))
    ok &= _line(torch.allclose(full[:, 5:], cache_ok), "RoPE decode step with correct offset matches full", True)
    ok &= _line(not torch.allclose(full[:, 5:], cache_bug), "RoPE decode step with offset=0 (bug) differs", True)
    cliff = learned_position_cliff()
    ok &= _line(cliff["acc_in_range"] == 1.0 and cliff["acc_beyond"] < 0.9, "learned absolute position cliff", cliff)
    b = alibi_bias(5, 2)
    ok &= _line(bool((b[:, 2, :3] <= 0).all()) and b[0, 4, 0] < b[0, 4, 3], "ALiBi penalty grows with distance", b[0, 4].tolist())
    for name, kw in (("plain", {}), ("interp x0.25", {"interp": 0.25}), ("ntk x4", {"ntk_scale": 4.0})):
        print(f"  info max RoPE angle at T=8192 d=64 [{name}]: {max_rope_angle(8192, 64, **kw):.1f} rad (trained T=2048: {max_rope_angle(2048, 64):.1f})")
    pad = torch.tensor([[True, True, False, False, False]])
    good, bug = position_ids_from_padding(pad), position_ids_from_padding(pad, bug=True)
    ok &= _line(good[0, 2:].tolist() == [0, 1, 2], "left-pad position ids (real tokens start at 0)", good.tolist())
    ok &= _line(bug[0, 2:].tolist() != [0, 1, 2], "buggy arange position ids", bug.tolist())
    return ok


def cmd_lora(_a) -> bool:
    from leaktest.checks import perturbation_leak_check
    from leaktest.model import build
    from tclab.lora import LoRALinear, apply_lora

    torch.manual_seed(0)
    ok = True
    print("[lora]")
    base = torch.nn.Linear(16, 8)
    x = torch.randn(4, 16)
    l = LoRALinear(base, r=4, alpha=8)
    with torch.no_grad():
        d0 = (l(x) - base(x)).abs().max().item()
    ok &= _line(d0 == 0.0, "zero-init B => identical to base, diff", d0)
    w0 = base.weight.clone()
    opt = torch.optim.SGD([p for p in l.parameters() if p.requires_grad], lr=0.1)
    for _ in range(3):
        opt.zero_grad()
        l(x).pow(2).sum().backward()
        opt.step()
    ok &= _line(torch.equal(w0, base.weight), "frozen base weight unchanged after 3 steps", True)
    with torch.no_grad():
        y = l(x)
        l.merge()
        scale = y.abs().max().item()
        dm = (l(x) - y).abs().max().item() / scale
        l.unmerge()
        du = (l(x) - y).abs().max().item() / scale
    ok &= _line(dm < 1e-5 and du < 1e-5, "merge / unmerge equivalence (relative diff)", f"{dm:.3e} / {du:.3e}")
    m = build("correct")
    wrapped = apply_lora(m, "qkv")
    for w in wrapped:
        torch.nn.init.normal_(w.adapters["default"]["B"], std=0.5)
    r = perturbation_leak_check(m, torch.randint(0, 64, (2, 10)), 64)
    ok &= _line(not r.leaked, f"LoRA on {len(wrapped)} qkv layers keeps causal mask", r.summary())
    return ok


CMDS = {"leak": cmd_leak, "encoder": cmd_encoder, "parallel": cmd_parallel, "positions": cmd_positions, "lora": cmd_lora}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tclab", description="Transformer Correctness Lab")
    p.add_argument("module", choices=[*CMDS, "all"])
    a = p.parse_args(argv)
    names = list(CMDS) if a.module == "all" else [a.module]
    results = {n: CMDS[n](a) for n in names}
    print("\nRESULT: " + ", ".join(f"{n}={'pass' if v else 'FAIL'}" for n, v in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
