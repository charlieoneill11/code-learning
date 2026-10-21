# Step 5 · The context gate

**You write:** `ContextGate.forward` in `src/engram/gate.py` (about 5 lines)
**Try it:** `make see N=5` · **Test:** `make step N=5`

## The idea

The memory vector e_t is a blind guess. It was found by hashing tokens, so it
might belong to a different phrase that collided, or the phrase might mean
something else here ("Apple" the fruit vs the company). The model's hidden
state h_t has already read the context. Engram lets the hidden state decide
how much of the memory to let in.

It works like a single attention score:

```
k = W_k e            what the memory is "about"     (the key)
v = W_v e            what it would add               (the value)
alpha = sigmoid( RMSNorm(h) . RMSNorm(k) / sqrt(d) )
output = alpha * v
```

**Why the RMSNorms?** After RMSNorm a vector has length sqrt(d). So the dot
product is d times the cosine of the angle between h and k, and dividing by
sqrt(d) leaves sqrt(d) times the cosine. Only the *direction* matters, not how
loud either vector is:

```
h and k point the same way     cos =  1   alpha = sigmoid( 4) = 0.98   (d = 16)
unrelated (orthogonal)         cos =  0   alpha = sigmoid( 0) = 0.50
opposite                       cos = -1   alpha = sigmoid(-4) = 0.02
```

Alpha is a single number per position: the memory is let in as a whole, or
shut out as a whole.

## Your task

`forward(h, e)`: return `(alpha * v, alpha)` with shapes (B, T, d) and
(B, T, 1). Use `self.W_k`, `self.W_v`, `self.norm_h` (on h) and `self.norm_k`
(on k).

## Quick guess before you code (optional)

In the step 8 experiment, where in a sentence do you expect alpha to be high?

## The tests check

- Agreeing memory gives sigmoid(sqrt(d)), opposite gives under 0.02, orthogonal gives exactly 0.5
- Output is exactly alpha times `W_v e`
- Scaling h by 100 doesn't change alpha
- Gradients reach both projections
- Exact agreement with the reference, with random norm weights

## Going further

Paper section 2.3, equations 3 and 4, and section 6.5, where the gate lights up
on multi-token names like "Alexander the Great". DeepSeek's demo code adds a
signed square root before the sigmoid. We follow the paper's equation.
