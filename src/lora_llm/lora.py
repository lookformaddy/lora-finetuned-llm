"""LoRA (Low-Rank Adaptation) implemented from scratch.

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language
Models" (arXiv:2106.09685). For a frozen pre-trained weight matrix
``W0`` of shape ``(out, in)``, LoRA learns a low-rank update

    W = W0 + (alpha / r) * (A @ B)^T ,   A in R^(in x r),  B in R^(r x out)

so the forward pass becomes ``y = W0 x + (alpha/r) * (x @ A @ B)``.
``A`` is initialised Gaussian, ``B`` is initialised to zero, so at the
start of training the adapted model is *exactly* the base model.

This module provides:
- ``LoRALinear``      — drop-in wrapper around ``nn.Linear``.
- ``inject_lora``     — replace targeted linear layers in a model.
- ``mark_only_lora_trainable`` — freeze the base, unfreeze adapters
  (plus optional "modules to save", PEFT-style).
- ``merge_lora``      — fold adapters into the base weights for export.
- ``save_adapter`` / ``load_adapter`` — adapter-only checkpoints
  (kilobytes, not megabytes — the operational point of LoRA).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    modules_to_save: tuple[str, ...] = ("lm_head",)

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    def to_dict(self) -> dict:
        d = asdict(self)
        d["target_modules"] = list(self.target_modules)
        d["modules_to_save"] = list(self.modules_to_save)
        d["scaling"] = self.scaling
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LoRAConfig":
        d = {k: v for k, v in d.items() if k != "scaling"}
        d["target_modules"] = tuple(d.get("target_modules", ("q_proj", "v_proj")))
        d["modules_to_save"] = tuple(d.get("modules_to_save", ("lm_head",)))
        return cls(**d)


class LoRALinear(nn.Module):
    """An ``nn.Linear`` with a parallel low-rank adapter branch."""

    def __init__(self, base: nn.Linear, cfg: LoRAConfig):
        super().__init__()
        if base.weight.ndim != 2:
            raise ValueError("LoRALinear expects a plain 2D linear layer")
        self.base = base
        self.cfg = cfg
        in_features, out_features = base.in_features, base.out_features
        self.in_features, self.out_features = in_features, out_features
        # A: (in, r) Gaussian; B: (r, out) zeros  =>  delta W == 0 at init.
        self.lora_A = nn.Parameter(torch.empty(in_features, cfg.rank))
        self.lora_B = nn.Parameter(torch.zeros(cfg.rank, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        # The wrapped base weight is frozen from the start.
        for p in self.base.parameters():
            p.requires_grad = False
        self.merged = False

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    def delta_weight(self) -> torch.Tensor:
        """The low-rank update in the (out, in) orientation of nn.Linear."""
        return self.cfg.scaling * (self.lora_A @ self.lora_B).t()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.merged:
            update = self.lora_dropout(x) @ self.lora_A @ self.lora_B
            out = out + self.cfg.scaling * update
        return out

    def merge(self) -> None:
        """Fold the adapter into the base weight in-place (idempotent)."""
        if self.merged:
            return
        with torch.no_grad():
            self.base.weight.data += self.delta_weight()
        self.merged = True

    def unmerge(self) -> None:
        if not self.merged:
            return
        with torch.no_grad():
            self.base.weight.data -= self.delta_weight()
        self.merged = False

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"r={self.cfg.rank}, alpha={self.cfg.alpha}, merged={self.merged}"
        )


# ----------------------------------------------------------------------
# Model surgery helpers
# ----------------------------------------------------------------------

def _set_module(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parent_name, _, attr = dotted_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, attr, new_module)


def _get_module(root: nn.Module, dotted_name: str) -> nn.Module:
    return root.get_submodule(dotted_name) if dotted_name else root


def inject_lora(model: nn.Module, cfg: LoRAConfig) -> list[str]:
    """Replace every ``nn.Linear`` whose leaf name is in
    ``cfg.target_modules`` with a ``LoRALinear``. Returns replaced names."""
    replaced: list[str] = []
    for name, module in list(model.named_modules()):
        leaf = name.rpartition(".")[2]
        if leaf in cfg.target_modules and isinstance(module, nn.Linear):
            _set_module(model, name, LoRALinear(module, cfg))
            replaced.append(name)
    if not replaced:
        raise ValueError(
            f"No modules matched target_modules={cfg.target_modules}"
        )
    return replaced


def lora_modules(model: nn.Module) -> dict[str, LoRALinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


def mark_only_lora_trainable(model: nn.Module, cfg: LoRAConfig) -> dict[str, int]:
    """Freeze everything, then unfreeze adapter params (+ modules_to_save).

    Returns a count summary used for the experiment report.
    """
    for p in model.parameters():
        p.requires_grad = False
    trainable = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
            trainable += module.lora_A.numel() + module.lora_B.numel()
    saved = 0
    for name in cfg.modules_to_save:
        try:
            module = _get_module(model, name)
        except AttributeError:
            continue
        for p in module.parameters():
            p.requires_grad = True
            saved += p.numel()
    return {"lora_params": trainable, "modules_to_save_params": saved}


def merge_lora(model: nn.Module) -> list[str]:
    """Merge all adapters in the model and swap LoRALinear wrappers back
    to plain ``nn.Linear`` layers. Returns merged module names."""
    merged_names = []
    for name, module in list(lora_modules(model).items()):
        module.merge()
        plain = nn.Linear(module.in_features, module.out_features,
                          bias=module.base.bias is not None)
        with torch.no_grad():
            plain.weight.copy_(module.base.weight)
            if module.base.bias is not None:
                plain.bias.copy_(module.base.bias)
        _set_module(model, name, plain)
        merged_names.append(name)
    return merged_names


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Only the adapter tensors — this is what gets saved per task."""
    state: dict[str, torch.Tensor] = {}
    for name, module in lora_modules(model).items():
        state[f"{name}.lora_A"] = module.lora_A.detach().cpu().clone()
        state[f"{name}.lora_B"] = module.lora_B.detach().cpu().clone()
    return state


def save_adapter(model: nn.Module, cfg: LoRAConfig, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "lora": adapter_state_dict(model),
        "modules_to_save": {},
    }
    # Persist modules_to_save weights alongside (PEFT behaviour) so a
    # reloaded adapter reproduces the tuned model exactly.
    for name in cfg.modules_to_save:
        try:
            module = _get_module(model, name)
        except AttributeError:
            continue
        payload["modules_to_save"][name] = {
            k: v.detach().cpu().clone() for k, v in module.state_dict().items()
        }
    torch.save(payload, out / "adapter.pt")
    (out / "adapter_config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2), encoding="utf-8"
    )


def load_adapter(model: nn.Module, adapter_dir: str | Path) -> LoRAConfig:
    """Load adapter weights into a model that already has LoRA injected
    (or inject them using the saved config if it has not)."""
    adapter_dir = Path(adapter_dir)
    cfg = LoRAConfig.from_dict(
        json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    )
    if not lora_modules(model):
        inject_lora(model, cfg)
    payload = torch.load(adapter_dir / "adapter.pt", map_location="cpu", weights_only=True)
    modules = lora_modules(model)
    for key, tensor in payload["lora"].items():
        mod_name, _, leaf = key.rpartition(".")
        module = modules[mod_name]
        param = module.lora_A if leaf == "lora_A" else module.lora_B
        with torch.no_grad():
            param.copy_(tensor)
    for name, state in payload.get("modules_to_save", {}).items():
        _get_module(model, name).load_state_dict(state)
    return cfg
