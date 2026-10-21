"""Step 3: hash an N-gram to a row.

There are far too many possible N-grams to give each its own row, so Engram
squeezes each one into a fixed-size table with a cheap hash: multiply each
token by its own odd constant, XOR the products together, and take the
remainder mod a prime table size.
"""

import numpy as np
import torch
from torch import Tensor


def ngram_hash(ngrams: Tensor, multipliers: Tensor, table_size: int) -> Tensor:
    """Hash each N-gram to a row in [0, table_size).

    ngrams: (..., n) int64 canonical IDs, oldest first, newest last
            (the layout suffix_ngrams returns).
    multipliers: (n,) odd int64. multipliers[0] multiplies the NEWEST token,
            multipliers[1] the one before it, and so on.
    Returns (...) int64 rows:
        mix = (newest * m[0]) XOR (previous * m[1]) XOR ...
        row = mix mod table_size

    Example: (the, Great) = (2, 11), multipliers (3, 5), 13 rows:
        11*3 XOR 2*5 = 33 XOR 10 = 43, and 43 mod 13 = 4.
    """
    # >>> step 3: your code
    raise NotImplementedError("step 3")
    # <<<


def make_multipliers(max_order: int, vocab_size: int, seed: int) -> Tensor:
    """Random odd multipliers, small enough that id * multiplier fits in int64.

    Same recipe as DeepSeek's demo code.
    """
    bound = max(1, (np.iinfo(np.int64).max // vocab_size) // 2)
    r = np.random.default_rng(seed).integers(0, bound, size=(max_order,), dtype=np.int64)
    return torch.from_numpy(r * 2 + 1)


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


def distinct_primes(at_least: int, count: int, taken: set[int]) -> list[int]:
    """The next `count` primes >= at_least that aren't in `taken`. Updates `taken`."""
    out, p = [], at_least
    while len(out) < count:
        if is_prime(p) and p not in taken:
            out.append(p)
            taken.add(p)
        p += 1
    return out
