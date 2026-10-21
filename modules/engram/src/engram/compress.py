"""Step 1: merge spellings.

A tokenizer gives "Apple", " apple" and "APPLE" three different IDs. Engram's
memory doesn't care about spelling, so before hashing, every token ID is mapped
to a canonical ID shared by all its spellings. That makes the memory table
smaller and lets every spelling of a phrase hit the same rows.
"""

import re
import unicodedata

import torch
from torch import Tensor


def normalize(text: str) -> str:
    """Return the canonical spelling of one token's text.

    Apply, in this order:
      1. Unicode NFKC, then NFD (splits "é" into "e" + a combining accent)
      2. drop combining accents ("é" -> "e")
      3. lowercase
      4. collapse each run of spaces, tabs and newlines into one space
      5. strip spaces from both ends, except that a token which is only
         whitespace becomes a single space " " (so it doesn't vanish)

    Examples: " Apple" -> "apple", "ÉCOLE" -> "ecole", "\\n\\n" -> " ".
    """
    # >>> step 1: your code
    s = unicodedata.normalize("NFKC", text)
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[ \t\r\n]+", " ", s)
    if s == " ":
        return s
    return s.strip()
    # <<<


def build_lookup(vocab: list[str]) -> tuple[Tensor, int]:
    """Map every token ID to a canonical ID.

    vocab[i] is the text of token i. Two tokens share a canonical ID exactly
    when their normalized text is equal. Canonical IDs are handed out in order
    of first appearance: 0 for vocab[0]'s spelling, then the next unused ID
    each time a new normalized spelling shows up. If normalizing gives an
    empty string, use the raw text as the key instead.

    Returns (lookup, n_canonical): lookup is an int64 tensor of shape
    (len(vocab),) with lookup[i] = canonical ID of token i.
    """
    # >>> step 1: your code
    key_to_id: dict[str, int] = {}
    lookup = []
    for text in vocab:
        key = normalize(text) or text
        if key not in key_to_id:
            key_to_id[key] = len(key_to_id)
        lookup.append(key_to_id[key])
    return torch.tensor(lookup, dtype=torch.long), len(key_to_id)
    # <<<


def compress(ids: Tensor, lookup: Tensor) -> Tensor:
    """Token IDs of any shape -> canonical IDs of the same shape."""
    return lookup[ids]
