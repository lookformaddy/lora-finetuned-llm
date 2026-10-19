from lora_llm.data import (
    CATEGORIES,
    build_dataset,
    generate_examples,
    load_split,
    split_examples,
)


def test_generation_is_deterministic():
    a = generate_examples(seed=13)
    b = generate_examples(seed=13)
    assert [x.to_json() for x in a] == [x.to_json() for x in b]
    assert len(a) > 500


def test_labels_are_valid():
    for ex in generate_examples(seed=1):
        assert ex.category in CATEGORIES
        assert ex.priority in {"P1", "P2", "P3", "P4"}
        assert ex.action  # non-empty runbook action


def test_response_format():
    ex = generate_examples(seed=1)[0]
    lines = ex.response.split("\n")
    assert lines[0] == f"Category: {ex.category}"
    assert lines[1] == f"Priority: {ex.priority}"
    assert lines[2].startswith("Action: ")
    assert ex.text == f"{ex.prompt}\n{ex.response}"
    assert ex.prompt.startswith("Ticket: ")


def test_splits_partition_the_pool():
    examples = generate_examples(seed=13)
    splits = split_examples(examples)
    total = sum(len(v) for v in splits.values())
    assert total == len(examples)
    # Partition by object identity (surface text may repeat across the
    # generated combinations; the split itself must never share a row).
    keys = {
        name: {id(ex) for ex in items} for name, items in splits.items()
    }
    assert keys["train"].isdisjoint(keys["val"])
    assert keys["train"].isdisjoint(keys["test"])
    assert keys["val"].isdisjoint(keys["test"])
    # rough proportions
    assert len(splits["train"]) > len(splits["val"])
    assert len(splits["test"]) > 50


def test_build_dataset_writes_files(tmp_path):
    counts = build_dataset(tmp_path)
    for name in ("train", "val", "test"):
        rows = load_split(tmp_path / f"{name}.jsonl")
        assert len(rows) == counts[name]
        assert set(rows[0]) >= {"instruction", "category", "priority", "action",
                                "prompt", "response", "text"}
    assert (tmp_path / "DATA_CARD.md").exists()
