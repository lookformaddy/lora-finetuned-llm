#!/usr/bin/env python3
"""Inference demo: base vs LoRA-tuned, side by side.

    python examples/inference_demo.py
    python examples/inference_demo.py --ticket "VPN drops every few minutes"

Requires artifacts from scripts/train.py. Shows, for the same ticket,
what the base model produces (generic prose drift, no format) versus the
LoRA-tuned model (the enterprise triage format it was taught).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_llm.evaluate import generate_response, parse_response  # noqa: E402
from lora_llm.lora import load_adapter  # noqa: E402
from lora_llm.train import load_base  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TICKETS = [
    "Hi team, VPN drops every few minutes when working from home in Ticket Desk. This is blocking our work.",
    "Urgent: dashboard totals do not match the source system in Finance Hub. Please advise.",
    "Hello support, laptop battery swells and the case will not close. Appreciate a quick fix.",
]


def show(model, tok, ticket: str, label: str) -> None:
    prompt = f"Ticket: {ticket}\nResponse:"
    out = generate_response(model, tok, prompt)
    parsed = parse_response(out)
    print(f"--- {label} ---")
    print(out.strip()[:500])
    print(f"[format compliant: {parsed is not None}]")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticket", help="custom ticket text (default: 3 built-in examples)")
    args = ap.parse_args()

    artifacts = ROOT / "artifacts"
    if not (artifacts / "base").exists():
        sys.exit("Artifacts not found — run `python scripts/train.py` first.")

    base, tok = load_base(artifacts / "base")
    tuned, _ = load_base(artifacts / "base")
    load_adapter(tuned, artifacts / "adapter")

    tickets = [args.ticket] if args.ticket else DEFAULT_TICKETS
    for ticket in tickets:
        print(f"=== Ticket: {ticket}\n")
        show(base, tok, ticket, "BASE MODEL")
        show(tuned, tok, ticket, "LoRA-TUNED MODEL")


if __name__ == "__main__":
    main()
