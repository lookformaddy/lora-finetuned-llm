# Model Card — LoRA-Tuned Support Triage Model (Toy Scale)

## Model details
- **Base model:** `TinyTransformer` — a decoder-only transformer written
  from scratch in this repo (d_model 128, 3 layers, 4 heads, 548,736
  parameters), pre-trained on synthetic generic business prose.
- **Adaptation:** LoRA implemented from scratch — rank 8, alpha 16
  (scaling 2.0), adapters on the attention `q_proj`/`v_proj` of every
  layer; LM head trained as a PEFT-style module-to-save. Trainable
  parameters during fine-tuning: 77,568 (14.1% of the model).
- **Artifacts:** base checkpoint (2.2 MB), adapter checkpoint (307 KB —
  49 KB of LoRA tensors + the saved LM head), merged model export (2.2 MB).
- **Framework:** PyTorch (CPU). No external pre-trained weights are used
  anywhere in this project.

## Intended use
- **Primary:** demonstrating and teaching the end-to-end LoRA workflow —
  dataset construction, frozen-base fine-tuning, adapter lifecycle
  (inject → train → save/load → merge), and base-vs-tuned evaluation.
- **Out of scope:** any real support triage. The model is a 0.55M-parameter
  teaching artifact trained on synthetic template data; it must not be
  used to route, prioritise, or answer real tickets.

## Training data
Fully synthetic (see `data/dataset/DATA_CARD.md`): 768 generated
enterprise support tickets (576 train / 96 validation / 96 test),
composed from 16 symptom templates × 8 applications with rule-based
category/priority/runbook-action labels. No real tickets, people, or
organisations. English only.

## Evaluation
Greedy decoding on the held-out test split (n = 96), whose
symptom×application combinations never appear in training:

| Metric | Base | LoRA-tuned |
|--------|-----:|-----------:|
| Format compliance | 0.00% | 100.00% |
| Category accuracy | 0.00% | 97.92% |
| Priority accuracy | 0.00% | 100.00% |
| Action token F1 | 0.00% | 93.34% |
| Exact response match | 0.00% | 32.29% |

Training: pre-train loss 6.24 → 2.96 (3 epochs, ~11 s); fine-tune train
loss 3.86 → 0.08, validation loss 2.66 → 0.09 (20 epochs, ~58 s) on CPU.
Full numbers: `reports/metrics.json`, `reports/experiment_report.md`,
`reports/training_log.csv`, `reports/loss_curve.svg`.

## Limitations
- **Template world:** the label mapping is rule-based by construction, so
  high scores show the workflow works — they do not indicate capability
  on real, ambiguous tickets. Real tickets are multi-intent, noisy, and
  context-dependent; expect large drops.
- **Exact match is the honest metric:** only 32% of generations reproduce
  the gold action verbatim; the rest are correct in structure and mostly
  correct in content (F1 0.93) with occasional entity substitutions.
- **Small-model failure modes:** on out-of-distribution phrasings the
  tuned model can pick a plausible-but-wrong category while keeping the
  format perfect — format compliance is not correctness.
- **No safety work:** no toxicity/bias evaluation, no PII handling, no
  human review loop. All of these are mandatory before any real use.
- **Language/domain:** English, IT-support register only.

## Ethical considerations
The dataset is synthetic, so no personal data is involved and no real
support worker's tickets were mined. If this workflow were applied to
real ticket corpora, data governance (consent, retention, PII redaction)
and a bias audit across requester groups would be required before
training, and automation of priority decisions would need a human
appeals path.
