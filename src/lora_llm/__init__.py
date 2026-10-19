"""lora_llm — a from-scratch LoRA fine-tuning pipeline for a small
transformer language model, implemented with PyTorch on CPU.

Modules:
    data        Synthetic enterprise support/ops instruction dataset builder.
    tokenizer   Word-level tokenizer built from the corpus.
    model       Small decoder-only transformer (the "base" model).
    lora        From-scratch LoRA adapters (low-rank, frozen base, merging).
    train       Training loops: base pre-training and LoRA fine-tuning.
    evaluate    Base-vs-tuned evaluation harness.
"""

__version__ = "0.1.0"
