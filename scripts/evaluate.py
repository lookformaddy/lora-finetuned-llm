#!/usr/bin/env python3
"""Standalone evaluation: base vs LoRA-tuned on the test split.

Requires artifacts produced by scripts/train.py. Prints the comparison
table and (re)writes reports/metrics_standalone.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_llm.evaluate import compare, evaluate_model, save_metrics  # noqa: E402
from lora_llm.lora import load_adapter  # noqa: E402
from lora_llm.train import load_base, load_records  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    data_dir = ROOT / "data" / "dataset"
    artifacts = ROOT / "artifacts"
    records = load_records(data_dir)

    base, tok = load_base(artifacts / "base")
    base_eval = evaluate_model(base, tok, records["test"])

    tuned, _ = load_base(artifacts / "base")
    load_adapter(tuned, artifacts / "adapter")
    tuned_eval = evaluate_model(tuned, tok, records["test"])

    print(compare(base_eval["metrics"], tuned_eval["metrics"]))
    save_metrics(
        {"base": base_eval["metrics"], "lora_tuned": tuned_eval["metrics"]},
        ROOT / "reports" / "metrics_standalone.json",
    )


if __name__ == "__main__":
    main()
