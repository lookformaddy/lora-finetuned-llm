"""Alignment regression tests for the fine-tuning batch plumbing.

The model predicts token t+1 from position t, so the collated targets
must be the labels shifted by one — an off-by-one here produces a model
with deceptively low teacher-forced loss that cannot generate.
"""
from lora_llm.tokenizer import Tokenizer
from lora_llm.train import _collate, encode_example

RECORD = {
    "prompt": "Ticket: vpn drops\nResponse:",
    "response": "Category: Network\nPriority: P2\nAction: Fix it.",
    "category": "Network",
    "priority": "P2",
    "action": "Fix it.",
    "instruction": "vpn drops",
    "text": "Ticket: vpn drops\nResponse:\nCategory: Network\nPriority: P2\nAction: Fix it.",
}


def _tok() -> Tokenizer:
    return Tokenizer.build([RECORD["text"]])


def test_encode_masks_prompt_and_includes_eos():
    tok = _tok()
    ids, labels = encode_example(tok, RECORD)
    assert ids[-1] == tok.eos_id
    # Prompt positions are masked; response positions carry their own id.
    prompt_len = len(tok.encode(RECORD["prompt"], add_bos=True, add_eos=False))
    assert all(lab == -1 for lab in labels[:prompt_len])
    assert labels[prompt_len:] == ids[prompt_len:]


def test_collate_shifts_targets_by_one():
    tok = _tok()
    ids, labels = encode_example(tok, RECORD)
    inputs, targets = _collate([(ids, labels)], pad_id=tok.pad_id)
    assert inputs.shape == targets.shape
    assert inputs[0].tolist() == ids[:-1]
    assert targets[0].tolist() == labels[1:]
    # Every supervised target is the *next* token of the original sequence.
    for t, tgt in enumerate(targets[0].tolist()):
        if tgt != -1:
            assert tgt == ids[t + 1]
