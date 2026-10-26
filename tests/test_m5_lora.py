import pytest
import torch

from leaktest.checks import perturbation_leak_check
from leaktest.model import build
from tclab.lora import LoRALinear, apply_lora


@pytest.fixture
def setup():
    torch.manual_seed(0)
    return torch.nn.Linear(16, 8), torch.randn(4, 16)


def _train(l, x, steps=3):
    opt = torch.optim.SGD([p for p in l.parameters() if p.requires_grad], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        l(x).pow(2).sum().backward()
        opt.step()


def test_zero_init_identical_to_base(setup):
    base, x = setup
    l = LoRALinear(base)
    assert torch.equal(l(x), base(x))


def test_random_b_init_bug_detected(setup):
    base, x = setup
    assert not torch.equal(LoRALinear(base, bug="random_b_init")(x), base(x))


def test_base_frozen_and_adapter_trains(setup):
    base, x = setup
    l = LoRALinear(base)
    w = base.weight.clone()
    _train(l, x)
    assert torch.equal(w, base.weight)
    assert l.adapters["default"]["B"].abs().sum() > 0


def test_merge_unmerge_equivalence(setup):
    base, x = setup
    l = LoRALinear(base)
    _train(l, x)
    w = base.weight.clone()
    with torch.no_grad():
        y = l(x)
        l.merge()
        assert torch.allclose(l(x), y, rtol=1e-4, atol=1e-5)
        l.unmerge()
        assert torch.allclose(l(x), y, rtol=1e-4, atol=1e-5)
        assert torch.allclose(base.weight, w, atol=1e-6)


def test_scaling_alpha_over_r(setup):
    base, x = setup
    l = LoRALinear(base, r=4, alpha=8)
    torch.nn.init.normal_(l.adapters["default"]["B"])
    a = l.adapters["default"]
    expected = base(x) + 2.0 * (x @ a["A"].T @ a["B"].T)
    assert torch.allclose(l(x), expected, atol=1e-6)
    bad = LoRALinear(base, r=4, alpha=8, bug="no_scaling")
    assert bad.scaling == 8 and l.scaling == 2


def test_multiple_adapters_switching(setup):
    base, x = setup
    l = LoRALinear(base)
    torch.nn.init.normal_(l.adapters["default"]["B"])
    y_default = l(x).detach()
    l.add_adapter("task_b")
    torch.nn.init.normal_(l.adapters["task_b"]["B"])
    y_b = l(x).detach()
    assert not torch.allclose(y_default, y_b)
    l.active = "default"
    assert torch.allclose(l(x), y_default)
    l.active = None
    assert torch.equal(l(x), base(x))


def test_lora_does_not_break_causal_mask():
    m = build("correct")
    for w in apply_lora(m, "qkv"):
        torch.nn.init.normal_(w.adapters["default"]["B"], std=0.5)
    assert not perturbation_leak_check(m, torch.randint(0, 64, (2, 10)), 64).leaked


def test_lora_on_buggy_model_still_detected():
    m = build("buggy_transpose")
    apply_lora(m, "qkv")
    assert perturbation_leak_check(m, torch.randint(0, 64, (2, 10)), 64).leaked


def test_only_adapter_params_trainable():
    m = build("correct")
    apply_lora(m, "qkv", r=2)
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    assert names and all("adapters" in n for n in names)
