# Step 3 · Hash an N-gram to a row

**You write:** `ngram_hash` in `src/engram/hashing.py` (about 5 lines)
**Try it:** `make see N=3` · **Test:** `make step N=3`

## The idea

With a 100k vocabulary there are 10^10 possible 2-grams and 10^15 3-grams.
You can't give each its own row. Engram's answer is the oldest trick in the
book: a **hash table**. Squash each N-gram into a row number in a table of
fixed size, and accept that some N-grams will share a row.

The hash is cheap and deterministic:

1. Multiply each token ID by its own odd constant (the newest token gets
   `m[0]`, the one before it `m[1]`, and so on)
2. XOR the products together
3. Take the remainder mod the table size (a prime)

Worked by hand, with canonical IDs the=2 and Great=11, multipliers (3, 5)
and 13 rows:

```
Great * 3  = 33   = 100001
the   * 5  = 10   = 001010
XOR        = 43   = 101011
43 mod 13  = 4    -> row 4
```

Swap the order and you get a different row: (Great, the) gives 6 XOR 55 = 49,
which is row 10. That's why each position gets its own multiplier.

Collisions happen. (of, Wales) = (4, 27) gives 81 XOR 20 = 69, and 69 mod 13 is
row 4 again. Step 4 deals with that.

## Your task

`ngram_hash(ngrams, multipliers, table_size)`: `ngrams` is (..., n) and oldest
first, the same layout step 2 returns. Return (...) int64 rows.
`torch.bitwise_xor` (or `^`) does XOR on integer tensors.

## Quick guess before you code (optional)

Why odd multipliers? What happens to the lowest bit of `x * m` if `m` is even?

## The tests check

- The three hand-worked examples above
- A 3-gram example
- Exact agreement with the reference on 9,600 random N-grams with real-size multipliers
- 200k random bigrams spread evenly over 1,009 rows

## Going further

Paper section 2.2 ("Multi-head hashing") and the demo code's
`NgramHashMapping._get_ngram_hashes`, which is this exact recipe in NumPy.
