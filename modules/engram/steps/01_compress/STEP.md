# Step 1 · Merge spellings

**You write:** `normalize` and `build_lookup` in `src/engram/compress.py` (about 15 lines)
**Try it:** `make see N=1` · **Test:** `make step N=1`

## The idea

Engram is a memory that a language model can look things up in. The lookup
key is the last few tokens, so "Alexander the Great" can fetch a stored vector
that says "Macedonian king, 4th century BC".

There's a snag before any of that. Tokenizers give different IDs to
`"Great"`, `" great"` and `"GREAT"`, so three spellings of the same phrase
would land in three different places in memory, and each would have to learn
the same fact separately. So the first thing Engram does is map every token
ID to a **canonical ID** shared by all its spellings:

```
token text     " Apple"   "apple"   " APPLE"   " pear"   "Pear"
token ID           1         2          3         4         5
canonical ID       1         1          1         2         2
```

You build this once, from the vocabulary, as a lookup table. At run time it's
a single indexing operation: `lookup[ids]`. In the paper this shrinks a
128k-token vocabulary by 23%.

## Your task

1. `normalize(text)`: lowercase, strip accents, tidy whitespace. The
   docstring gives the exact recipe and order. Python's `unicodedata` module
   does the Unicode parts.
2. `build_lookup(vocab)`: give each new normalized spelling the next free ID,
   and return the table as an int64 tensor.

## Quick guess before you code (optional)

The step 8 experiment's vocabulary has 282 tokens. Its name words come in
three casings (`zuto`, `Zuto`, `ZUTO`). How many canonical IDs do you expect?

## The tests check

- Casing, accents, look-alike Unicode characters and whitespace all fold together
- A whitespace-only token becomes a single space instead of vanishing
- IDs are handed out in order of first appearance
- The step 8 vocabulary shrinks from 282 to 202

## Going further

Paper section 2.2, "Tokenizer compression". The DeepSeek demo code uses
HuggingFace `tokenizers` normalizers for the same recipe.
