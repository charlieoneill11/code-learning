# Step 8 · Train it

**You write:** `Block.forward` in `src/engram/model.py` (about 5 lines), then run the experiment
**Run:** `make experiment` (about 3 minutes; `make experiment QUICK=1` takes about 1) · **Test:** `make step N=8`

## The idea

Time to see whether any of this helps. `model.py` has a tiny decoder-only
transformer (2 layers, d_model = 64), and `data.py` has a synthetic world
built for exactly what Engram is meant to be good at:

```
... blor kima  Zuto Rafe Mebi  tesh  vuna  dazo ...
               \____name_____/ \fact/
```

There are 1,500 entities. Each has a 2- or 3-word name and one fact word that
always follows it. The name words are shared by many entities, so no single
word predicts the fact. Only the whole N-gram does. Each mention is randomly
written as `zuto`, `Zuto` or `ZUTO`, and those are different token IDs.

To predict the fact, a normal transformer has to use attention to gather the
name's tokens, then memorize 1,500 mappings in its MLP weights. Engram can
store each fact directly in the row its N-gram hashes to.

## Your task

`Block.forward(h, ids)`: a standard pre-norm block, with Engram's update added
**first**, before attention, when this block has one:

```
h = h + engram(h, ids)        # only if self.engram is not None
h = h + attn(norm1(h))
h = h + mlp(norm2(h))
```

Then run `make experiment`. It trains four models with the same backbone and
seed:

| Model | What it tests |
| --- | --- |
| backbone | no Engram |
| wider backbone | the same parameter count as the Engram model, spent on bigger MLPs instead |
| backbone + Engram | Engram in layer 1 |
| Engram, no merging | without step 1, so the three casings are separate N-grams |

Then it probes the trained Engram model.

## Quick guesses before you run it (optional)

1. Which model learns the facts first? Does the wider backbone beat the plain one?
2. What happens to fact accuracy if you switch Engram off after training?
3. Where in a sentence will the gate (alpha) be open?

## What you should see

Reference numbers. Yours should match closely, since everything is seeded.

```
model                 params    @250    @750   @1500
backbone                151k    3.0%   15.9%   79.2%
wider backbone          300k    2.7%    8.6%   62.5%
backbone + Engram       299k   99.6%   99.8%  100.0%
Engram, no merging      299k   86.3%   99.6%  100.0%
```

- Engram knows the facts after 250 steps. The backbones need well over 1,500,
  and spending the same extra parameters on bigger MLPs makes learning slower,
  not faster.
- Without merging spellings the start is slower, because each casing has to
  learn its own rows.
- Switching Engram off after training drops fact accuracy to about 2%. The
  facts live in the table, not in the backbone.
- The gate stays shut (0.00) on filler and fact words, and opens on name words.
  It's low on a name's first word, whose N-grams are mostly filler, and near 1
  by the last. Nobody told it which words are names. This is the paper's
  section 6.5 figure in miniature.

## Try next (optional)

`uv run python -m engram.experiment --quick` takes flags. Two worth running:

- `--rows 101`: tables 10x smaller, so lots of collisions. Engram still reaches
  about 99%, just more slowly.
- `--rows 101 --heads 1`: the same small table with a single hash head. Engram
  stalls at about 20%, which is step 4's lesson made concrete.

## The tests check

- A block without Engram is the two standard residual updates
- With Engram, its update comes first, and it gets h and the token IDs
- A 200-step run on a small world learns more than 80% of the facts

## Going further

Other things to try: `--entities 5000`, putting Engram in layer 0, or giving
the wider backbone 3x the steps. The paper's version of this analysis is
section 6.1, which uses logit lens and CKA. Engram's early layers do work that
the MoE baseline only finishes several layers later.
