#!/usr/bin/env python3
"""Build the synthetic instruction dataset (JSONL splits + data card)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_llm.data import build_dataset  # noqa: E402

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/dataset"
    counts = build_dataset(out)
    print(f"Dataset written to {out}: {counts}")
