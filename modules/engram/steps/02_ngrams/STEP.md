# Step 2 · Suffix N-grams

**You write:** `suffix_ngrams` in `src/engram/ngrams.py` (about 3 lines)
**Try it:** `make see N=2` · **Test:** `make step N=2`

## The idea

What should the model look up at each position? Engram uses the phrase that
**ends** at that position: the last 2 tokens (a 2-gram) and the last 3 tokens
(a 3-gram). At "Great" in "Only Alexander the Great":

```
position      0       1           2      3
token         Only    Alexander   the    Great
2-gram at 3:                      the    Great
3-gram at 3:          Alexander   the    Great
```

Only looking backwards keeps the model causal: the memory at position t never
peeks at tokens the model hasn't seen yet. At the start of a sequence there
aren't enough earlier tokens, so the missing slots are filled with a pad ID.

You'll compute this for every position at once, as a tensor. A good mental
picture: stack shifted copies of the sequence.

```
ids                [ 5  7  2 11]
shifted by 1       [ 0  5  7  2]
shifted by 2       [ 0  0  5  7]
```

Reading down a column gives that position's N-gram, newest at the top.

## Your task

`suffix_ngrams(ids, n, pad_id)`: (B, T) in, (B, T, n) out, **oldest token
first** along the last axis. `torch.nn.functional.pad` and `torch.stack` are
all you need, though there are several other ways.

## Quick guess before you code (optional)

What should `suffix_ngrams(ids, 1, pad_id)` return?

## The tests check

- The docstring example and the Alexander example
- The pad ID fills slots before the start
- Rows in a batch don't leak into each other
- Changing later tokens never changes earlier N-grams (causality)

## Going further

Paper section 2.2: Engram uses N = 2 and 3. Larger N means rarer, more
specific phrases, and many more of them to store.
