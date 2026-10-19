"""A small word-level tokenizer.

The corpus is narrow (synthetic enterprise support text plus a general
word list used in base pre-training), so a word-level vocabulary is
sufficient, transparent, and fast on CPU. Punctuation is split into its
own tokens so the model sees the response format markers (`Category:`,
`Priority:`, `Action:`) as stable units.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|\n|[^\sA-Za-z0-9]")

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


class Tokenizer:
    def __init__(self, vocab: list[str]):
        self.itos = list(vocab)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    # -- construction -----------------------------------------------------
    @classmethod
    def build(cls, texts: list[str], min_freq: int = 1, max_vocab: int = 4096) -> "Tokenizer":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenize(text))
        words = [
            w
            for w, c in counter.most_common()
            if c >= min_freq and w not in SPECIALS
        ]
        vocab = SPECIALS + sorted(words[: max_vocab - len(SPECIALS)])
        return cls(vocab)

    # -- basics -----------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids = [self.stoi.get(tok, self.stoi[UNK]) for tok in tokenize(text)]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_specials: bool = True) -> str:
        tokens = []
        for i in ids:
            tok = self.itos[i]
            if skip_specials and tok in SPECIALS:
                continue
            tokens.append(tok)
        text = " ".join(tokens)
        # Re-attach punctuation and newlines for readability.
        text = re.sub(r" ([.,;:!?%])", r"\1", text)
        text = text.replace(" \n ", "\n").replace(" \n", "\n").replace("\n ", "\n")
        return text

    # -- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.itos, indent=0), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
