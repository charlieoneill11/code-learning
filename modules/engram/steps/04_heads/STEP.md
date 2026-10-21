# Step 4 · Many heads, one table

**You write:** `NgramMemory.row_ids` and `NgramMemory.forward` in `src/engram/memory.py` (about 8 lines)
**Try it:** `make see N=4` · **Test:** `make step N=4`

## The idea

In step 3, (the, Great) and (of, Wales) both landed on row 4. With a single
hash they'd share that row forever, and the stored vector would be a muddle
of both.

The fix is to hash each N-gram several times, into several tables (**heads**)
of different prime sizes. Collisions in different heads are independent, so
two N-grams that collide in one head almost never collide in all of them:

```
               head 0 (13 rows)   head 1 (17 rows)   head 2 (19 rows)
(the, Great)   43 mod 13 = 4      43 mod 17 = 9      43 mod 19 = 5
(of, Wales)    69 mod 13 = 4      69 mod 17 = 1      69 mod 19 = 12
```

They share head 0's row but differ in heads 1 and 2. The memory vector for a
position, e_t, is every head's row **concatenated**, so the two phrases still
get different vectors.

**One table, offsets.** Rather than one `nn.Embedding` per head, all heads
share one big table. Head j owns a block of rows starting at `offsets[j]`:

```
rows 0..12  -> head 0    (offset 0)
rows 13..29 -> head 1    (offset 13)
rows 30..48 -> head 2    (offset 30)
```

So (the, Great) reads global rows 4, 13 + 9 = 22 and 30 + 5 = 35. One table
means one lookup call, which matters later when the table moves off the GPU
(step 9).

Engram does this for each N-gram order (2 and 3), so with 4 heads per order a
position reads 8 rows.

## Your task

Everything is set up in `__init__`: `self.orders`, `self.table_sizes[i][j]`,
`self.multipliers`, `self.offsets` and `self.table`.

1. `row_ids(cids)`: (B, T) to (B, T, n_heads) global rows. Use your
   `suffix_ngrams` and `ngram_hash`. Order n uses `self.multipliers[:n]`.
   Columns go order by order: all heads of order 2, then all heads of order 3.
2. `forward(cids)`: look the rows up and concatenate them into (B, T, n_heads * head_dim).

## Quick guess before you code (optional)

With 1 head of ~211 rows, about 1 in 211 pairs of different bigrams collide.
Roughly how often do they collide in all 4 heads?

## The tests check

- The toy rows (4, 9, 5) and global rows (4, 22, 35)
- (of, Wales) collides with (the, Great) in head 0 only
- Column order is orders, then heads
- `forward` returns each head's row, side by side, and gradients reach the table
- Exact agreement with the reference
- 4 heads give at least 20x fewer full collisions than 1 head

## Going further

Paper section 2.2, "Multi-Head Hashing". The real model uses 8 heads per order
and tables of about 650k rows each, and it keeps every prime distinct across
heads and layers.
