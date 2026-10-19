from lora_llm.tokenizer import Tokenizer, tokenize

TEXTS = [
    "Category: Access & Identity\nPriority: P2\nAction: Unlock the account.",
    "The weekly review covered the dashboard and the rota.",
]


def test_roundtrip():
    tok = Tokenizer.build(TEXTS)
    for text in TEXTS:
        ids = tok.encode(text)
        assert tok.decode(ids) == text


def test_specials_present_and_first():
    tok = Tokenizer.build(TEXTS)
    assert tok.itos[:4] == ["<pad>", "<bos>", "<eos>", "<unk>"]
    assert tok.pad_id == 0


def test_unknown_words_map_to_unk():
    tok = Tokenizer.build(TEXTS)
    ids = tok.encode("zzqwxy unknown", add_bos=False, add_eos=False)
    assert all(i == tok.stoi["<unk>"] for i in ids)


def test_newline_is_a_token():
    assert "\n" in tokenize("a\nb")
