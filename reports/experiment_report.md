# Experiment Report — LoRA Fine-Tuning of a Small Transformer

Seed 42 · CPU-only · dataset {'train': 576, 'val': 96, 'test': 96} · vocab 510 tokens.

## Setup

- Base model: decoder-only transformer, 548,736 parameters (d_model=128, layers=3, heads=4).
- Pre-training: 3 epochs on generic business prose (11.2s). Final logged loss 2.9564.
- LoRA: rank r=8, alpha=16.0, scaling=2.0, targets=['q_proj', 'v_proj'], modules_to_save=['lm_head'].
- Trainable parameters during fine-tuning: 77,568 (14.1% of the model) — 12,288 adapter + 65,280 head.
- Fine-tuning: 20 epochs, loss on response tokens only (57.9s). Train loss 3.8633 → 0.0834; val loss 2.6552 → 0.0875.
- Adapter checkpoint: 307.3 KB vs merged model 2161 KB.

## Results — base vs LoRA-tuned

Evaluated on the held-out test split (n = 96, greedy decoding).

| Metric | Base model | LoRA-tuned | Delta |
|--------|-----------:|-----------:|------:|
| Format compliance | 0.00% | 100.00% | +100.00% |
| Category accuracy | 0.00% | 97.92% | +97.92% |
| Priority accuracy | 0.00% | 100.00% | +100.00% |
| Action token F1 | 0.00% | 93.34% | +93.34% |
| Exact response match | 0.00% | 32.29% | +32.29% |

Merge/reload checks: the merged model and a fresh base + reloaded adapter reproduce the tuned metrics exactly (asserted in the pipeline). Merged modules: blocks.0.attn.q_proj, blocks.0.attn.v_proj, blocks.1.attn.q_proj, blocks.1.attn.v_proj, blocks.2.attn.q_proj, blocks.2.attn.v_proj.

## Sample generations (test split)

### Example 1
**Ticket:** Hi, Vpn drops every few minutes when working from home in learning portal

**Gold:** Category: Network · Priority: P2

**Base:** the morning and the morning.

**LoRA-tuned:** 
Category: Network
Priority: P2
Action: Reinstall the VPN client, switch the tunnel to the backup gateway, and monitor the session for drops.

### Example 2
**Ticket:** Hi team, New starter needs access to the shared mailbox in inventory manager Appreciate a quick fix.

**Gold:** Category: Access & Identity · Priority: P4

**Base:** the morning.

**LoRA-tuned:** 
Category: Access & Identity
Priority: P4
Action: Raise an access request for Supply Chain approval, then grant mailbox delegation in the mail admin console.

### Example 3
**Ticket:** Hi, Account is locked after too many failed attempts in asset tracker

**Gold:** Category: Access & Identity · Priority: P3

**Base:** the morning and the morning and the morning.

**LoRA-tuned:** 
Category: Access & Identity
Priority: P3
Action: Unlock the account in the identity provider and coach the lockout policy.
