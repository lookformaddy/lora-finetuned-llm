"""A small decoder-only transformer language model (PyTorch, CPU-friendly).

This is the frozen "base" model for the LoRA demo. It is deliberately tiny
(~0.6M parameters) so the full pipeline — pre-training, LoRA fine-tuning,
evaluation — runs in minutes on a laptop CPU, while still containing the
real components LoRA targets in production models: multi-head causal
self-attention with separate Q/K/V/O projections and MLP blocks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 128
    n_layer: int = 3
    n_head: int = 4
    d_ff: int = 256
    max_len: int = 160
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        # Separate projections (named for LoRA targeting: q_proj / v_proj).
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, d = x.shape
        q = self.q_proj(x).view(bsz, seq, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq, self.n_head, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.attn_dropout.p if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seq, d)
        return self.o_proj(out)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        h = self.ln2(x)
        h = self.fc2(F.gelu(self.fc1(h)))
        return x + self.dropout(h)


class TinyTransformer(nn.Module):
    """Decoder-only LM with untied input embedding and LM head.

    The head is intentionally untied: during LoRA fine-tuning the head is
    a PEFT-style "module to save" so tokens that are rare in base
    pre-training (e.g. the `Category:` format markers) can still be
    learned, while every base transformer weight stays frozen.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        bsz, seq = idx.shape
        if seq > self.cfg.max_len:
            raise ValueError(f"sequence length {seq} exceeds max_len {self.cfg.max_len}")
        pos = torch.arange(seq, device=idx.device)
        x = self.tok_embed(idx) + self.pos_embed(pos).unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 64,
        eos_id: int | None = None,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """Greedy (temperature=0) or sampled generation for a single row."""
        self.eval()
        out = idx.clone()
        for _ in range(max_new_tokens):
            window = out[:, -self.cfg.max_len :]
            logits, _ = self(window)
            next_logits = logits[:, -1, :]
            if temperature and temperature > 0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            else:
                nxt = next_logits.argmax(dim=-1, keepdim=True)
            out = torch.cat([out, nxt], dim=1)
            if eos_id is not None and int(nxt.item()) == eos_id:
                break
        return out

    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad or not trainable_only
        )
