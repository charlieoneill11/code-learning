"""Step 4: many heads, one table.

One hash means unlucky N-grams share a row forever. So each N-gram order gets
several hash heads, each with its own prime table size. Two N-grams that
collide in one head almost never collide in all of them, and the model can
tell them apart from the combination.

All heads for all orders live in ONE nn.Embedding. Head j owns a contiguous
block of rows starting at offsets[j], so head j's row r is global row
offsets[j] + r. The rows a token reads are concatenated into its memory
vector e_t.
"""

import torch
from torch import Tensor, nn

from .hashing import distinct_primes, make_multipliers, ngram_hash
from .ngrams import suffix_ngrams


class NgramMemory(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        orders: tuple[int, ...] = (2, 3),
        heads: int = 4,
        min_rows: int = 1009,
        head_dim: int = 16,
        pad_id: int = 0,
        seed: int = 0,
        taken_primes: set[int] | None = None,
        table_sizes: list[list[int]] | None = None,
        multipliers: Tensor | None = None,
    ):
        """vocab_size: number of canonical IDs (after step 1).
        orders: which N-gram lengths to use. heads: hash heads per order.
        min_rows: each head's table size is a distinct prime >= this.
        table_sizes / multipliers: set them explicitly (tests use small toy values).
        """
        super().__init__()
        self.orders = tuple(orders)
        self.pad_id = pad_id
        self.head_dim = head_dim
        if table_sizes is None:
            taken = taken_primes if taken_primes is not None else set()
            table_sizes = [distinct_primes(min_rows, heads, taken) for _ in self.orders]
        if multipliers is None:
            multipliers = make_multipliers(max(self.orders), vocab_size, seed)
        # table_sizes[i][j] = size of head j for order self.orders[i]
        self.table_sizes = [list(s) for s in table_sizes]
        flat = [size for sizes in self.table_sizes for size in sizes]
        offsets = [0]
        for size in flat[:-1]:
            offsets.append(offsets[-1] + size)
        self.register_buffer("multipliers", multipliers.long(), persistent=False)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long), persistent=False)
        self.n_heads = len(flat)
        self.out_dim = self.n_heads * head_dim
        self.table = nn.Embedding(sum(flat), head_dim)

    def row_ids(self, cids: Tensor) -> Tensor:
        """Global table rows read at each position.

        cids: (B, T) canonical token IDs.
        Returns (B, T, n_heads) int64. Column order: every head of
        self.orders[0], then every head of self.orders[1], and so on. Order n
        uses self.multipliers[:n], and head j adds self.offsets[j].
        """
        # >>> step 4: your code
        cols = []
        for i, n in enumerate(self.orders):
            grams = suffix_ngrams(cids, n, self.pad_id)
            for size in self.table_sizes[i]:
                cols.append(ngram_hash(grams, self.multipliers[:n], size))
        return torch.stack(cols, dim=-1) + self.offsets
        # <<<

    def forward(self, cids: Tensor) -> Tensor:
        """Memory vector e_t at each position.

        cids: (B, T) canonical token IDs.
        Returns (B, T, n_heads * head_dim): each head's row, concatenated in
        the same column order as row_ids.
        """
        # >>> step 4: your code
        return self.table(self.row_ids(cids)).flatten(-2)
        # <<<
