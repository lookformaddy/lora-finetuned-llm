"""Evaluation harness: base model vs LoRA-tuned model on the held-out set.

Metrics (all computed on greedy generations from the test split):

- **Format compliance** — the generation contains `Category:`,
  `Priority:`, and `Action:` fields, in that order.
- **Category accuracy** — parsed category == gold category.
- **Priority accuracy** — parsed priority == gold priority.
- **Action token F1** — token-overlap F1 between the generated action
  and the gold runbook action (exact match is too brittle a bar for a
  0.6M-parameter model; F1 shows partial credit honestly).
- **Exact response match** — the whole formatted response is identical.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from .model import TinyTransformer
from .tokenizer import Tokenizer

_FIELD_RE = re.compile(
    r"Category:\s*(?P<category>.+?)\s*Priority:\s*(?P<priority>P[1-4])\s*Action:\s*(?P<action>.+)",
    re.DOTALL,
)


def parse_response(text: str) -> dict | None:
    """Parse the enterprise response format; None if non-compliant."""
    m = _FIELD_RE.search(text)
    if not m:
        return None
    return {
        "category": m.group("category").strip().split("\n")[0].strip(),
        "priority": m.group("priority").strip(),
        "action": " ".join(m.group("action").split()).strip(),
    }


def _token_f1(pred: str, gold: str) -> float:
    pred_tokens = re.findall(r"[a-z0-9]+", pred.lower())
    gold_tokens = re.findall(r"[a-z0-9]+", gold.lower())
    if not pred_tokens or not gold_tokens:
        return 0.0
    from collections import Counter

    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


@torch.no_grad()
def generate_response(model: TinyTransformer, tok: Tokenizer, prompt: str,
                      max_new_tokens: int = 80) -> str:
    ids = tok.encode(prompt, add_bos=True, add_eos=False)
    window = ids[-model.cfg.max_len :]
    out = model.generate(
        torch.tensor([window]), max_new_tokens=max_new_tokens, eos_id=tok.eos_id
    )
    new_ids = out[0, len(window) :].tolist()
    return tok.decode(new_ids)


def evaluate_model(model: TinyTransformer, tok: Tokenizer,
                   records: list[dict], max_new_tokens: int = 80) -> dict:
    """Run the harness over records (gold fields come from the dataset)."""
    results = {
        "n": len(records),
        "format_compliant": 0,
        "category_correct": 0,
        "priority_correct": 0,
        "action_f1_sum": 0.0,
        "exact_match": 0,
    }
    samples: list[dict] = []
    for rec in records:
        generated = generate_response(model, tok, rec["prompt"], max_new_tokens)
        parsed = parse_response(generated)
        gold_action = " ".join(rec["action"].split())
        entry = {
            "instruction": rec["instruction"],
            "gold_category": rec["category"],
            "gold_priority": rec["priority"],
            "generated": generated,
            "parsed": parsed,
        }
        if parsed is not None:
            results["format_compliant"] += 1
            if parsed["category"] == rec["category"]:
                results["category_correct"] += 1
            if parsed["priority"] == rec["priority"]:
                results["priority_correct"] += 1
            results["action_f1_sum"] += _token_f1(parsed["action"], gold_action)
            gold_response = " ".join(rec["response"].split())
            if " ".join(generated.split()) == gold_response:
                results["exact_match"] += 1
        samples.append(entry)
    n = max(results["n"], 1)
    metrics = {
        "n": results["n"],
        "format_compliance": round(results["format_compliant"] / n, 4),
        "category_accuracy": round(results["category_correct"] / n, 4),
        "priority_accuracy": round(results["priority_correct"] / n, 4),
        "action_token_f1": round(results["action_f1_sum"] / n, 4),
        "exact_match": round(results["exact_match"] / n, 4),
    }
    return {"metrics": metrics, "samples": samples}


def compare(base_metrics: dict, tuned_metrics: dict) -> str:
    """Render the headline comparison table (markdown)."""
    rows = [
        ("Format compliance", "format_compliance"),
        ("Category accuracy", "category_accuracy"),
        ("Priority accuracy", "priority_accuracy"),
        ("Action token F1", "action_token_f1"),
        ("Exact response match", "exact_match"),
    ]
    lines = [
        f"Evaluated on the held-out test split (n = {base_metrics['n']}, greedy decoding).",
        "",
        "| Metric | Base model | LoRA-tuned | Delta |",
        "|--------|-----------:|-----------:|------:|",
    ]
    for label, key in rows:
        b, t = base_metrics[key], tuned_metrics[key]
        lines.append(f"| {label} | {b:.2%} | {t:.2%} | {t - b:+.2%} |")
    return "\n".join(lines)


def save_metrics(metrics: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def loss_curve_svg(log: list[dict], path: str | Path, width: int = 720, height: int = 260) -> None:
    """Render the training-loss curve as a dependency-free SVG."""
    series = {}
    for row in log:
        series.setdefault(row["phase"], []).append((row["step"], row["loss"]))
    all_losses = [l for pts in series.values() for _, l in pts]
    if not all_losses:
        return
    lo, hi = min(all_losses), max(all_losses)
    span = (hi - lo) or 1.0
    max_step = max(s for pts in series.values() for s, _ in pts) or 1
    colours = {"pretrain": "#888888", "finetune": "#2563eb"}
    polylines = []
    for phase, pts in series.items():
        coords = " ".join(
            f"{30 + (s / max_step) * (width - 50):.1f},{height - 30 - ((l - lo) / span) * (height - 60):.1f}"
            for s, l in pts
        )
        polylines.append(
            f'<polyline fill="none" stroke="{colours.get(phase, "#333")}" '
            f'stroke-width="1.6" points="{coords}"/>'
        )
    legend = "".join(
        f'<text x="{40 + i * 130}" y="16" font-size="11" fill="{colours.get(p, "#333")}">■ {p}</text>'
        for i, p in enumerate(series)
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<rect width="100%" height="100%" fill="white"/>'
        f"{legend}{''.join(polylines)}</svg>"
    )
    Path(path).write_text(svg, encoding="utf-8")
