"""Tests for the from-scratch LoRA implementation: shapes, the zero-init
identity, merge correctness, frozen-base guarantees, and save/load."""
import torch
import torch.nn as nn

from lora_llm.lora import (
    LoRAConfig,
    LoRALinear,
    adapter_state_dict,
    inject_lora,
    load_adapter,
    lora_modules,
    mark_only_lora_trainable,
    merge_lora,
    save_adapter,
)
from lora_llm.model import ModelConfig, TinyTransformer


def _model() -> TinyTransformer:
    torch.manual_seed(0)
    return TinyTransformer(
        ModelConfig(vocab_size=64, d_model=32, n_layer=2, n_head=4, d_ff=64, max_len=32)
    )


def test_adapter_shapes():
    layer = LoRALinear(nn.Linear(20, 12), LoRAConfig(rank=4, alpha=8.0))
    assert layer.lora_A.shape == (20, 4)   # (in, r)
    assert layer.lora_B.shape == (4, 12)  # (r, out)
    assert layer.delta_weight().shape == (12, 20)  # (out, in), like W


def test_zero_init_is_identity():
    """B starts at zero, so an untrained adapter must not change outputs."""
    torch.manual_seed(0)
    base = nn.Linear(16, 16)
    layer = LoRALinear(base, LoRAConfig(rank=4))
    x = torch.randn(3, 16)
    assert torch.allclose(layer(x), base(x), atol=1e-6)


def test_scaling_formula():
    cfg = LoRAConfig(rank=8, alpha=16.0)
    assert cfg.scaling == 2.0
    layer = LoRALinear(nn.Linear(8, 8), cfg)
    with torch.no_grad():
        layer.lora_A.copy_(torch.randn(8, 8))
        layer.lora_B.copy_(torch.randn(8, 8))
    expected = cfg.scaling * (layer.lora_A @ layer.lora_B).t()
    assert torch.allclose(layer.delta_weight(), expected)


def test_merge_correctness():
    """Merged plain linear must produce identical outputs to the adapter."""
    torch.manual_seed(1)
    base = nn.Linear(24, 24)
    layer = LoRALinear(base, LoRAConfig(rank=4, alpha=4.0))
    with torch.no_grad():  # simulate a trained adapter
        layer.lora_B.copy_(torch.randn_like(layer.lora_B))
    x = torch.randn(5, 24)
    before = layer(x)
    w0 = layer.base.weight.detach().clone()
    merged_weight = w0 + layer.delta_weight()
    layer.merge()
    after = layer(x)
    assert torch.allclose(before, after, atol=1e-5)
    assert torch.allclose(layer.base.weight, merged_weight, atol=1e-6)


def test_inject_targets_only_q_and_v():
    model = _model()
    cfg = LoRAConfig(rank=4)
    replaced = inject_lora(model, cfg)
    assert len(replaced) == 2 * 2  # q_proj + v_proj, x2 layers
    assert all(name.endswith(("q_proj", "v_proj")) for name in replaced)
    modules = lora_modules(model)
    assert len(modules) == 4
    # k/o projections and MLP stay plain Linear
    assert isinstance(model.blocks[0].attn.k_proj, nn.Linear)
    assert not isinstance(model.blocks[0].attn.k_proj, LoRALinear)


def test_frozen_base_and_trainable_counts():
    model = _model()
    cfg = LoRAConfig(rank=4, modules_to_save=("lm_head",))
    inject_lora(model, cfg)
    counts = mark_only_lora_trainable(model, cfg)
    # LoRA params: per targeted layer r*(in+out) = 4*(32+32) = 256, x4 layers
    assert counts["lora_params"] == 4 * 4 * (32 + 32)
    assert counts["modules_to_save_params"] == 64 * 32  # lm_head weight
    for name, module in lora_modules(model).items():
        assert not module.base.weight.requires_grad, name
        assert module.lora_A.requires_grad and module.lora_B.requires_grad
    assert not model.blocks[0].fc1.weight.requires_grad
    assert model.lm_head.weight.requires_grad  # module to save


def test_base_weights_do_not_move_during_training():
    torch.manual_seed(2)
    model = _model()
    cfg = LoRAConfig(rank=4)
    inject_lora(model, cfg)
    mark_only_lora_trainable(model, cfg)
    frozen_before = {
        n: p.detach().clone() for n, p in model.named_parameters()
        if not p.requires_grad
    }
    adapter_before = {
        n: p.detach().clone() for n, p in model.named_parameters()
        if p.requires_grad and "lora_" in n
    }
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.5)
    x = torch.randint(0, 64, (2, 10))
    _, loss = model(x, targets=x)
    opt.zero_grad()
    loss.backward()
    # Frozen params must receive no gradient at all.
    for n, p in model.named_parameters():
        if not p.requires_grad:
            assert p.grad is None, n
    opt.step()
    for n, p in model.named_parameters():
        if n in frozen_before:
            assert torch.equal(p, frozen_before[n]), f"frozen param moved: {n}"
    moved = [n for n, p in model.named_parameters()
             if n in adapter_before and not torch.equal(p, adapter_before[n])]
    assert moved, "expected at least one adapter tensor to update"


def test_merge_lora_swaps_back_to_plain_linear():
    model = _model()
    cfg = LoRAConfig(rank=4)
    inject_lora(model, cfg)
    for module in lora_modules(model).values():
        with torch.no_grad():
            module.lora_B.copy_(torch.randn_like(module.lora_B))
    x = torch.randint(0, 64, (2, 8))
    model.eval()
    with torch.no_grad():
        before, _ = model(x)
    merged = merge_lora(model)
    assert len(merged) == 4
    assert not lora_modules(model)  # wrappers gone
    with torch.no_grad():
        after, _ = model(x)
    assert torch.allclose(before, after, atol=1e-5)


def test_adapter_save_load_roundtrip(tmp_path):
    torch.manual_seed(3)
    model = _model()
    cfg = LoRAConfig(rank=4, alpha=8.0)
    inject_lora(model, cfg)
    mark_only_lora_trainable(model, cfg)
    for module in lora_modules(model).values():
        with torch.no_grad():
            module.lora_B.copy_(torch.randn_like(module.lora_B))
    with torch.no_grad():
        model.lm_head.weight.add_(0.01)  # trained module-to-save
    save_adapter(model, cfg, tmp_path)
    assert (tmp_path / "adapter.pt").exists()
    assert set(adapter_state_dict(model)) == {
        f"{n}.{leaf}"
        for n in lora_modules(model)
        for leaf in ("lora_A", "lora_B")
    }

    # _model() seeds internally, so `fresh` starts from identical base
    # weights; after load_adapter it must reproduce `model` exactly.
    fresh = _model()
    loaded_cfg = load_adapter(fresh, tmp_path)
    assert loaded_cfg.rank == cfg.rank and loaded_cfg.alpha == cfg.alpha
    for key, tensor in adapter_state_dict(model).items():
        assert torch.equal(adapter_state_dict(fresh)[key], tensor)
    assert torch.allclose(fresh.lm_head.weight, model.lm_head.weight, atol=1e-6)
    x = torch.randint(0, 64, (2, 8))
    model.eval(); fresh.eval()
    with torch.no_grad():
        a, _ = model(x)
        b, _ = fresh(x)
    assert torch.allclose(a, b, atol=1e-6)
