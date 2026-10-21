"""A synthetic world full of multi-token names, for the step 8 experiment.

Text is random filler words with entity mentions dropped in. Each entity has a
name made of 2 or 3 "name words" and one fact word that always follows it:

    ... blor kima  Zuto Rafe Mebi  tesh  vuna ...
                   \\_____name____/ \\fact/

Name words are shared between many entities, so no single word predicts the
fact: only the whole N-gram does. That's exactly the kind of lookup Engram is
built for. Each mention is written in a random casing (Zuto / zuto / ZUTO),
and each casing is a different token ID, so step 1's compression matters too.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

PAD, BOS = 0, 1
CASINGS = (str.lower, str.title, str.upper)


@dataclass
class WorldConfig:
    n_entities: int = 3000
    n_name_words: int = 40
    n_fact_words: int = 64
    n_filler_words: int = 96
    two_word_share: float = 0.25  # fraction of entities with 2-word names
    mention_rate: float = 0.35  # chance that the next item is a mention, not filler
    seq_len: int = 64
    seed: int = 0


class World:
    def __init__(self, cfg: WorldConfig = WorldConfig()):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        words = _make_words(cfg.n_name_words + cfg.n_fact_words + cfg.n_filler_words, rng)
        name_words = words[: cfg.n_name_words]
        fact_words = words[cfg.n_name_words : cfg.n_name_words + cfg.n_fact_words]
        filler_words = words[cfg.n_name_words + cfg.n_fact_words :]

        # Vocabulary: <pad>, <bos>, then " word" tokens. Name words come in 3 casings.
        self.vocab = ["<pad>", "<bos>"]
        self.name_ids = [[_add(self.vocab, " " + c(w)) for c in CASINGS] for w in name_words]
        self.fact_ids = [_add(self.vocab, " " + w) for w in fact_words]
        self.filler_ids = [_add(self.vocab, " " + w) for w in filler_words]
        # kind[token_id]: "special", "name", "fact" or "filler"
        self.kind = ["special"] * 2 + ["name"] * (3 * cfg.n_name_words)
        self.kind += ["fact"] * cfg.n_fact_words + ["filler"] * cfg.n_filler_words

        # Entities: unique names. No 2-word name is also the start or the end of a
        # 3-word name, so a name's last word always says unambiguously that the
        # fact comes next.
        names: set[tuple[int, ...]] = set()
        n_two = int(cfg.n_entities * cfg.two_word_share)
        while len(names) < n_two:
            names.add(tuple(rng.choice(cfg.n_name_words, size=2, replace=False).tolist()))
        tails = set(names)
        while len(names) < cfg.n_entities:
            name = tuple(rng.choice(cfg.n_name_words, size=3, replace=False).tolist())
            if name[1:] not in tails and name[:2] not in tails:
                names.add(name)
        self.names = sorted(names)
        self.facts = rng.integers(0, cfg.n_fact_words, size=len(self.names)).tolist()

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[Tensor, Tensor]:
        """Returns (ids, is_fact): ids (B, seq_len + 1) int64 starting with <bos>;
        is_fact (B, seq_len + 1) bool, True where the token is an entity's fact."""
        T = self.cfg.seq_len + 1
        ids = np.zeros((batch_size, T), dtype=np.int64)
        is_fact = np.zeros((batch_size, T), dtype=bool)
        for b in range(batch_size):
            row, fact_mask = [BOS], [False]
            while len(row) < T:
                if rng.random() < self.cfg.mention_rate:
                    e = int(rng.integers(len(self.names)))
                    casing = int(rng.integers(len(CASINGS)))
                    row += [self.name_ids[w][casing] for w in self.names[e]]
                    fact_mask += [False] * len(self.names[e])
                    row.append(self.fact_ids[self.facts[e]])
                    fact_mask.append(True)
                else:
                    row.append(self.filler_ids[int(rng.integers(len(self.filler_ids)))])
                    fact_mask.append(False)
            ids[b], is_fact[b] = row[:T], fact_mask[:T]
        return torch.from_numpy(ids), torch.from_numpy(is_fact)

    def decode(self, ids) -> str:
        return "".join(self.vocab[int(i)] for i in ids)


def _add(vocab: list[str], token: str) -> int:
    vocab.append(token)
    return len(vocab) - 1


def _make_words(n: int, rng: np.random.Generator) -> list[str]:
    consonants, vowels = "bdfgklmnprstvz", "aeiou"
    syllables = [c + v for c in consonants for v in vowels]
    out: list[str] = []
    seen: set[str] = set()
    while len(out) < n:
        w = "".join(rng.choice(syllables, size=2))
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out
