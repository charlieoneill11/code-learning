# Step 6 · The short convolution

**You write:** `ShortConv.forward` in `src/engram/conv.py` (about 4 lines)
**Try it:** `make see N=6` · **Test:** `make step N=6`

## The idea

Each position fetched its memory on its own. A tiny convolution lets nearby
positions' gated memories mix, which widens what each position can see and
adds a nonlinearity:

```
Y = SiLU( Conv1D( RMSNorm(V) ) ) + V        (V = the gated values from step 5)
```

This step is the `SiLU(Conv1D(RMSNorm(V)))` part. Step 7 adds the `+ V`.

Three details make it this particular convolution:

- **Causal.** Output t may only use inputs at t and earlier. Pad on the left
  only, by (kernel_size - 1) * dilation, and nothing on the right.
- **Dilated.** Kernel size 4, dilation 3 (the longest N-gram order), so
  position t reads t, t-3, t-6 and t-9: one tap per earlier 3-gram block.
- **Depthwise.** Each channel has its own 4 weights and channels never mix
  (`groups=dim`, already set up).

```
input   t-9  .  .  t-6  .  .  t-3  .  .  t
          \          \          \        |
           w0         w1         w2      w3   -> output t
```

The weights start at **zero**, so at initialization this branch outputs
exactly 0 and Engram is just the gated memory. Training turns it on only if
it helps.

## Your task

`forward(x)`: (B, T, dim) in, (B, T, dim) out. `nn.Conv1d` wants
(B, channels, T), so transpose in and back out. `F.pad` does the left padding.

## Quick guess before you code (optional)

What would go wrong if you used `padding=` inside `nn.Conv1d` instead of
padding by hand?

## The tests check

- Output is exactly 0 at initialization
- Changing future inputs never changes past outputs
- Kernel weight j reads input t - 3 * (3 - j)
- Channels don't mix
- Exact agreement with the reference

## Going further

Paper section 2.3, equation 5. Ablations in section 6.2 show removing it hurts
only a little. Zero init keeps the module an identity at the start.
