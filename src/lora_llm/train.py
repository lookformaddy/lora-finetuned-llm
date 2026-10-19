"""Training: base pre-training on generic business prose, then LoRA
fine-tuning on the synthetic instruction dataset.

Design intent — mirror the real enterprise workflow at toy scale:
1. **Pre-train** a small transformer on generic business prose so it
   "knows the language" of the domain but has never seen the task format.
2. **Fine-tune with LoRA**: freeze the base, train only low-rank adapters
   (plus the LM head as a PEFT-style module-to-save) on instruction data
   with the loss masked to response tokens.
3. The base model scores ~0 on format compliance; the tuned model learns
   the triage format and the ticket -> decision mapping. That delta is
   the measurable effect of the adapters.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from .data import DEPARTMENTS, load_split
from .lora import LoRAConfig, inject_lora, mark_only_lora_trainable
from .model import ModelConfig, TinyTransformer
from .tokenizer import Tokenizer

# ----------------------------------------------------------------------
# Generic pre-training corpus (prose only — never the task format)
# ----------------------------------------------------------------------

_PROSE_TEMPLATES = [
    "The weekly operations review covered {n} open items across the {dept} team.",
    "When a request arrives, the service desk records its category, estimates its priority, and assigns an action to the right team.",
    "The {dept} team published the monthly summary and shared it with the leadership group.",
    "Every change follows the same path: proposal, review, approval, scheduling, and a short note on the outcome.",
    "The dashboard refresh finished at {h} in the morning and the figures were checked against the source system.",
    "A short post-incident note explains what happened, what was done, and what will prevent a repeat.",
    "The onboarding guide walks new starters through the tools, the teams, and the escalation paths.",
    "Capacity planning looks at demand from {dept}, the current backlog, and the hiring plan for the quarter.",
    "The vendor review compared cost, support quality, and the roadmap for the next two releases.",
    "Meeting notes from the planning session list the decisions, the owners, and the dates for each action.",
    "The quarterly audit sampled records from {dept} and found the process in good shape overall.",
    "A runbook describes the symptom, the likely cause, the checks to run, and the point at which to escalate.",
    "The service catalogue explains what each team offers, the expected response time, and how to ask for help.",
    "Budget holders in {dept} review spending monthly and flag anything unusual to finance.",
    "The training session covered the new workflow, the common mistakes, and where to find help.",
    "Good documentation is short, current, and written for the person who reads it at two in the morning.",
    "The migration plan moves one team at a time, with a rollback window after each step.",
    "Customer feedback this month praised response times and asked for clearer status updates.",
    "The rota ensures cover for the support desk during holidays and busy periods.",
    "Each project closes with a retrospective: what worked, what did not, and what changes next time.",
]


def general_corpus(seed: int = 7, sentences: int = 1600) -> list[str]:
    rng = random.Random(seed)
    out = []
    for _ in range(sentences):
        tpl = rng.choice(_PROSE_TEMPLATES)
        out.append(
            tpl.format(
                n=rng.randint(2, 48),
                h=rng.randint(5, 11),
                dept=rng.choice(DEPARTMENTS),
            )
        )
    return out


# ----------------------------------------------------------------------
# Config & data plumbing
# ----------------------------------------------------------------------

@dataclass
class TrainConfig:
    seed: int = 42
    d_model: int = 128
    n_layer: int = 3
    n_head: int = 4
    d_ff: int = 256
    max_len: int = 160
    pretrain_epochs: int = 3
    pretrain_lr: float = 3e-3
    pretrain_batch: int = 32
    block_size: int = 96
    finetune_epochs: int = 20
    finetune_lr: float = 3e-3
    finetune_batch: int = 16
    lora: LoRAConfig = field(default_factory=LoRAConfig)


def encode_example(tok: Tokenizer, record: dict) -> tuple[list[int], list[int]]:
    """(input_ids, labels) with loss only on the response tokens."""
    prompt_ids = tok.encode(record["prompt"], add_bos=True, add_eos=False)
    resp_ids = tok.encode("\n" + record["response"], add_bos=False, add_eos=False)
    resp_ids = resp_ids + [tok.eos_id]
    input_ids = prompt_ids + resp_ids
    labels = [-1] * len(prompt_ids) + resp_ids
    return input_ids, labels


def _collate(batch: list[tuple[list[int], list[int]]], pad_id: int):
    """Shift for next-token prediction: the model at position t predicts
    token t+1, so inputs are ids[:-1] and targets are labels[1:], where
    labels are aligned with ids (labels[t] == ids[t] on response tokens,
    -1 on the prompt). Padding goes on the right; target pad is -1."""
    maxlen = max(len(ids) - 1 for ids, _ in batch)
    inputs, labels = [], []
    for ids, lab in batch:
        inp, tgt = ids[:-1], lab[1:]
        pad = maxlen - len(inp)
        inputs.append(inp + [pad_id] * pad)
        labels.append(tgt + [-1] * pad)
    return torch.tensor(inputs), torch.tensor(labels)


def _batch_loss(model, batch, pad_id, device="cpu") -> torch.Tensor:
    inputs, labels = _collate(batch, pad_id)
    _, loss = model(inputs.to(device), targets=labels.to(device))
    return loss


@torch.no_grad()
def response_loss(model, encoded: list[tuple[list[int], list[int]]], pad_id: int,
                  batch_size: int = 32) -> float:
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(encoded), batch_size):
        loss = _batch_loss(model, encoded[i : i + batch_size], pad_id)
        total += float(loss) * len(encoded[i : i + batch_size])
        count += len(encoded[i : i + batch_size])
    model.train()
    return total / max(count, 1)


# ----------------------------------------------------------------------
# Phase 1: base pre-training (plain LM on prose)
# ----------------------------------------------------------------------

def pretrain_base(model: TinyTransformer, tok: Tokenizer, cfg: TrainConfig,
                  log: list[dict]) -> TinyTransformer:
    torch.manual_seed(cfg.seed)
    stream: list[int] = []
    for sentence in general_corpus(seed=cfg.seed):
        stream.extend(tok.encode(sentence, add_bos=False, add_eos=True))
    block = cfg.block_size + 1
    chunks = [stream[i : i + block] for i in range(0, len(stream) - block, block)]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.pretrain_lr, weight_decay=0.01)
    rng = random.Random(cfg.seed)
    step = 0
    model.train()
    for epoch in range(cfg.pretrain_epochs):
        rng.shuffle(chunks)
        for i in range(0, len(chunks), cfg.pretrain_batch):
            batch = chunks[i : i + cfg.pretrain_batch]
            inputs = torch.tensor([c[:-1] for c in batch])
            targets = torch.tensor([c[1:] for c in batch])
            _, loss = model(inputs, targets=targets)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 10 == 0 or step == 1:
                log.append({"phase": "pretrain", "step": step, "epoch": epoch + 1,
                            "loss": round(loss.item(), 4)})
    return model


# ----------------------------------------------------------------------
# Phase 2: LoRA fine-tuning (loss on response tokens only)
# ----------------------------------------------------------------------

def finetune_lora(model: TinyTransformer, tok: Tokenizer,
                  train_records: list[dict], val_records: list[dict],
                  cfg: TrainConfig, log: list[dict]) -> dict:
    inject_lora(model, cfg.lora)
    counts = mark_only_lora_trainable(model, cfg.lora)
    train_enc = [encode_example(tok, r) for r in train_records]
    val_enc = [encode_example(tok, r) for r in val_records]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.finetune_lr, weight_decay=0.0)
    rng = random.Random(cfg.seed + 1)
    step = 0
    model.train()
    for epoch in range(cfg.finetune_epochs):
        rng.shuffle(train_enc)
        epoch_loss, batches = 0.0, 0
        for i in range(0, len(train_enc), cfg.finetune_batch):
            loss = _batch_loss(model, train_enc[i : i + cfg.finetune_batch], tok.pad_id)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            epoch_loss += loss.item()
            batches += 1
        val = response_loss(model, val_enc, tok.pad_id)
        log.append({"phase": "finetune", "step": step, "epoch": epoch + 1,
                    "loss": round(epoch_loss / max(batches, 1), 4),
                    "val_loss": round(val, 4)})
    return counts


# ----------------------------------------------------------------------
# Checkpoint helpers
# ----------------------------------------------------------------------

def save_base(model: TinyTransformer, tok: Tokenizer, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_config": model.cfg.to_dict(), "state_dict": model.state_dict()},
        out / "base_model.pt",
    )
    tok.save(out / "tokenizer.json")


def load_base(base_dir: str | Path) -> tuple[TinyTransformer, Tokenizer]:
    base_dir = Path(base_dir)
    blob = torch.load(base_dir / "base_model.pt", map_location="cpu", weights_only=True)
    model = TinyTransformer(ModelConfig(**blob["model_config"]))
    model.load_state_dict(blob["state_dict"])
    tok = Tokenizer.load(base_dir / "tokenizer.json")
    return model, tok


def build_tokenizer(train_records: list[dict]) -> Tokenizer:
    texts = list(general_corpus())
    for r in train_records:
        texts.append(r["text"])
    return Tokenizer.build(texts)


def load_records(data_dir: str | Path) -> dict[str, list[dict]]:
    data_dir = Path(data_dir)
    return {name: load_split(data_dir / f"{name}.jsonl")
            for name in ("train", "val", "test")}
