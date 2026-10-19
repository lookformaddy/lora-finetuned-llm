#!/usr/bin/env python3
"""End-to-end pipeline: build data -> pre-train base -> evaluate base ->
LoRA fine-tune -> evaluate tuned -> save adapter + merged model -> report.

Everything runs on CPU in a few minutes with the default config.

Usage:
    python scripts/train.py [--finetune-epochs N] [--pretrain-epochs N]
                            [--rank R] [--seed S]
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from lora_llm import data as data_mod  # noqa: E402
from lora_llm.evaluate import compare, evaluate_model, loss_curve_svg, save_metrics  # noqa: E402
from lora_llm.lora import LoRAConfig, load_adapter, merge_lora, save_adapter  # noqa: E402
from lora_llm.model import TinyTransformer, ModelConfig  # noqa: E402
from lora_llm.train import (  # noqa: E402
    TrainConfig,
    build_tokenizer,
    finetune_lora,
    load_base,
    load_records,
    pretrain_base,
    save_base,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "dataset"))
    ap.add_argument("--artifacts-dir", default=str(ROOT / "artifacts"))
    ap.add_argument("--reports-dir", default=str(ROOT / "reports"))
    ap.add_argument("--pretrain-epochs", type=int, default=3)
    ap.add_argument("--finetune-epochs", type=int, default=20)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    artifacts = Path(args.artifacts_dir)
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)

    # 1. Data ------------------------------------------------------------
    counts = data_mod.build_dataset(args.data_dir)
    records = load_records(args.data_dir)
    tok = build_tokenizer(records["train"])
    print(f"Dataset: {counts} | vocab: {tok.vocab_size}")

    cfg = TrainConfig(
        seed=args.seed,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        lora=LoRAConfig(rank=args.rank, alpha=args.alpha),
    )
    model_cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        d_model=cfg.d_model,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        d_ff=cfg.d_ff,
        max_len=cfg.max_len,
    )
    log: list[dict] = []

    # 2. Pre-train base --------------------------------------------------
    t0 = time.time()
    base = TinyTransformer(model_cfg)
    total_params = base.num_parameters()
    pretrain_base(base, tok, cfg, log)
    pretrain_secs = time.time() - t0
    save_base(base, tok, artifacts / "base")
    print(f"Base pre-trained in {pretrain_secs:.1f}s | params: {total_params:,}")

    # 3. Evaluate base ---------------------------------------------------
    base_eval = evaluate_model(base, tok, records["test"])
    print(f"Base metrics: {base_eval['metrics']}")

    # 4. LoRA fine-tune (from a pristine copy of the base weights) -------
    tuned, _ = load_base(artifacts / "base")
    t1 = time.time()
    ft_counts = finetune_lora(tuned, tok, records["train"], records["val"], cfg, log)
    finetune_secs = time.time() - t1
    print(f"LoRA fine-tuned in {finetune_secs:.1f}s | trainable: {ft_counts}")

    tuned_eval = evaluate_model(tuned, tok, records["test"])
    print(f"Tuned metrics: {tuned_eval['metrics']}")

    # 5. Save adapter, then merged model ---------------------------------
    save_adapter(tuned, cfg.lora, artifacts / "adapter")
    adapter_bytes = (artifacts / "adapter" / "adapter.pt").stat().st_size
    merged_names = merge_lora(tuned)
    save_base(tuned, tok, artifacts / "merged")
    merged_bytes = (artifacts / "merged" / "base_model.pt").stat().st_size

    # Sanity: merged model must reproduce tuned metrics exactly.
    merged_model, _ = load_base(artifacts / "merged")
    merged_eval = evaluate_model(merged_model, tok, records["test"])
    assert merged_eval["metrics"] == tuned_eval["metrics"], (
        "Merged model metrics differ from adapter model — merge bug!"
    )

    # Sanity: adapter reload reproduces tuned metrics exactly.
    reload_model, _ = load_base(artifacts / "base")
    load_adapter(reload_model, artifacts / "adapter")
    reload_eval = evaluate_model(reload_model, tok, records["test"])
    assert reload_eval["metrics"] == tuned_eval["metrics"], (
        "Reloaded adapter metrics differ — save/load bug!"
    )

    # 6. Reports ----------------------------------------------------------
    with (reports / "training_log.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["phase", "step", "epoch", "loss", "val_loss"])
        writer.writeheader()
        for row in log:
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})
    loss_curve_svg(log, reports / "loss_curve.svg")
    all_metrics = {
        "dataset": counts,
        "vocab_size": tok.vocab_size,
        "total_params": total_params,
        "trainable": ft_counts,
        "adapter_bytes": adapter_bytes,
        "merged_model_bytes": merged_bytes,
        "pretrain_seconds": round(pretrain_secs, 1),
        "finetune_seconds": round(finetune_secs, 1),
        "base": base_eval["metrics"],
        "lora_tuned": tuned_eval["metrics"],
        "merged": merged_eval["metrics"],
    }
    save_metrics(all_metrics, reports / "metrics.json")
    _write_report(reports / "experiment_report.md", cfg, all_metrics, log,
                  base_eval, tuned_eval, merged_names)
    print(f"Reports written to {reports}")


def _write_report(path: Path, cfg: TrainConfig, m: dict, log: list[dict],
                  base_eval: dict, tuned_eval: dict, merged_names: list[str]) -> None:
    ft_log = [r for r in log if r["phase"] == "finetune"]
    pt_log = [r for r in log if r["phase"] == "pretrain"]
    trainable_total = m["trainable"]["lora_params"] + m["trainable"]["modules_to_save_params"]
    lines = [
        "# Experiment Report — LoRA Fine-Tuning of a Small Transformer",
        "",
        f"Seed {cfg.seed} · CPU-only · dataset {m['dataset']} · vocab {m['vocab_size']} tokens.",
        "",
        "## Setup",
        "",
        f"- Base model: decoder-only transformer, {m['total_params']:,} parameters "
        f"(d_model={cfg.d_model}, layers={cfg.n_layer}, heads={cfg.n_head}).",
        f"- Pre-training: {cfg.pretrain_epochs} epochs on generic business prose "
        f"({m['pretrain_seconds']}s). Final logged loss {pt_log[-1]['loss'] if pt_log else 'n/a'}.",
        f"- LoRA: rank r={cfg.lora.rank}, alpha={cfg.lora.alpha}, scaling={cfg.lora.scaling}, "
        f"targets={list(cfg.lora.target_modules)}, modules_to_save={list(cfg.lora.modules_to_save)}.",
        f"- Trainable parameters during fine-tuning: {trainable_total:,} "
        f"({trainable_total / m['total_params']:.1%} of the model) — "
        f"{m['trainable']['lora_params']:,} adapter + {m['trainable']['modules_to_save_params']:,} head.",
        f"- Fine-tuning: {cfg.finetune_epochs} epochs, loss on response tokens only "
        f"({m['finetune_seconds']}s). Train loss {ft_log[0]['loss']} → {ft_log[-1]['loss']}; "
        f"val loss {ft_log[0].get('val_loss')} → {ft_log[-1].get('val_loss')}.",
        f"- Adapter checkpoint: {m['adapter_bytes'] / 1024:.1f} KB vs merged model "
        f"{m['merged_model_bytes'] / 1024:.0f} KB.",
        "",
        "## Results — base vs LoRA-tuned",
        "",
        compare(m["base"], m["lora_tuned"]),
        "",
        "Merge/reload checks: the merged model and a fresh base + reloaded adapter "
        "reproduce the tuned metrics exactly (asserted in the pipeline). "
        f"Merged modules: {', '.join(merged_names)}.",
        "",
        "## Sample generations (test split)",
        "",
    ]
    for i in range(min(3, len(tuned_eval["samples"]))):
        b = base_eval["samples"][i]
        t = tuned_eval["samples"][i]
        lines += [
            f"### Example {i + 1}",
            f"**Ticket:** {t['instruction']}",
            "",
            f"**Gold:** Category: {t['gold_category']} · Priority: {t['gold_priority']}",
            "",
            f"**Base:** {b['generated'][:300]}",
            "",
            f"**LoRA-tuned:** {t['generated'][:300]}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
