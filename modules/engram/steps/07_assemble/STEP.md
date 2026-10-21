# Step 7 · Assemble Engram

**You write:** `Engram.forward` in `src/engram/engram.py` (about 6 lines)
**Try it:** `make see N=7` · **Test:** `make step N=7`

## The idea

You now have every piece. Here's the path one position takes through Engram:

```
raw token IDs ──compress──> canonical IDs ──memory──> e_t
                (step 1)                  (steps 2-4)
                                                         \
hidden state h_t ─────────────────────────────────────> gate ──> v = alpha * W_v e
                                                        (step 5)         │
                                                                         ▼
                                              update = v + ShortConv(v)
                                                       (step 6)
```

Engram doesn't change h itself. It returns an **update**, and the transformer
block adds it: `h = h + engram(h, ids)` (that's step 8). Two things are worth
noticing:

- The memory path depends only on **token IDs**, not on h. Step 9 exploits this.
- Only the gate looks at h. That's where context gets a say.

## Your task

`forward(h, ids)`: return `v + self.conv(v)`, shape (B, T, d_model). Also save
the gate's alpha in `self.last_alpha` (detached), so the step 8 experiment
can see where the gate opens. `compress`, `self.lookup`, `self.memory`,
`self.gate` and `self.conv` are all ready.

## Quick guess before you code (optional)

At initialization the conv is zero. What is the update then, in terms of the
gate?

## The tests check

- Output shape, and `last_alpha` is stored with no gradient
- Changing future tokens or hidden states never changes earlier outputs
- "the Great", "the great" and "the GREAT" give identical outputs, thanks to step 1
- The conv branch is included
- Exact agreement with the reference

## Going further

Paper section 2.3 and Figure 1. The real model also has 4 parallel residual
branches ("mHC", section 2.4). Each branch gets its own W_k and gate, while
the table and W_v are shared. We use one branch, which is the "w/o multi
branch" ablation in section 6.2.
