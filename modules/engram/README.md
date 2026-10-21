# Engram: memory you look up instead of compute

Paper: [Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](https://arxiv.org/abs/2601.07372) (DeepSeek-AI and Peking University, 2026).
Official demo code: [deepseek-ai/Engram](https://github.com/deepseek-ai/Engram).
Visual explainer: [Engram, step by step](https://claude.ai/artifact/TN7uQrX7iu4fjXXsqPZPDK).

**The idea in one paragraph.** A transformer has to *compute* that "Alexander
the Great" is one thing, rebuilding it from three tokens with attention and
MLPs every time. Engram gives the model a giant hash table keyed by the last 2
and 3 tokens, so it can *look up* a vector for the phrase in O(1). A gate lets
the context decide whether the memory fits. In the paper, swapping some MoE
experts for Engram memory at the same total parameters and compute improves
knowledge benchmarks (MMLU +3.4) and, surprisingly, reasoning even more (BBH +5.0).

## What you'll build

| # | Step | You write | Lines |
| --- | --- | --- | --- |
| 1 | [Merge spellings](steps/01_compress/STEP.md) | `normalize`, `build_lookup` | ~15 |
| 2 | [Suffix N-grams](steps/02_ngrams/STEP.md) | `suffix_ngrams` | ~3 |
| 3 | [Hash to a row](steps/03_hash/STEP.md) | `ngram_hash` | ~5 |
| 4 | [Many heads, one table](steps/04_heads/STEP.md) | `NgramMemory.row_ids`, `.forward` | ~8 |
| 5 | [The context gate](steps/05_gate/STEP.md) | `ContextGate.forward` | ~5 |
| 6 | [The short convolution](steps/06_conv/STEP.md) | `ShortConv.forward` | ~4 |
| 7 | [Assemble Engram](steps/07_assemble/STEP.md) | `Engram.forward` | ~6 |
| 8 | [Train it](steps/08_train/STEP.md) | `Block.forward`, then run the experiment | ~5 |
| 9 | [Prefetch from host memory](steps/09_prefetch/STEP.md) (stretch) | `PrefetchedMemory`, `prefetch_all` | ~10 |

Each step builds on the ones before, so do them in order.

## Where things live

```
src/engram/
  compress.py  ngrams.py  hashing.py  memory.py      steps 1-4: the lookup
  gate.py  conv.py  engram.py                         steps 5-7: the module
  model.py                                            step 8: tiny transformer (you wire Engram in)
  prefetch.py                                         step 9
  data.py  train.py  experiment.py  see.py            plumbing: synthetic world, training, demos
steps/NN_name/   STEP.md + tests
fixtures/        golden outputs from the reference, for exact-match tests
tools/           make_fixtures.py
```
