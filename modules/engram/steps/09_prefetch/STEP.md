# Step 9 (stretch) · Prefetch from host memory

**You write:** `PrefetchedMemory.start`, `PrefetchedMemory.forward` and `prefetch_all` in `src/engram/prefetch.py` (about 10 lines)
**Try it:** `make see N=9` · **Test:** `make step N=9`

## The idea

The paper's biggest Engram table has 100 billion parameters, far more than
fits on a GPU. So it lives in ordinary CPU RAM ("host memory"), and rows are
copied over the PCIe bus when needed. That copy is slow. Done naively, every
Engram layer would stall the GPU while its rows crawl across.

But you noticed in step 7 that the rows a layer reads depend **only on the
token IDs**, and those are known before the forward pass starts. So:

```
naive:      [layer 0][layer 1: wait for fetch.....][layer 2][layer 3]...
prefetch:   [fetch for layer 1 ........]
            [layer 0][layer 1][layer 2][layer 3]...      fetch hid behind layer 0
```

Work out every Engram layer's rows up front, start all the copies in the
background, then run the model. By the time a layer needs its rows, they've
arrived. The paper measures under 3% slowdown with a 100B table in host memory.
MoE routing can't do this, because which expert a token picks depends on its
hidden state.

Here the "GPU" is just your model, and `HostTable` stands in for host memory.
`gather(rows)` returns `table[rows]` after a simulated transfer delay, and logs
when each fetch started.

## Your task

1. `PrefetchedMemory.start(cids)`: compute rows with `self.memory.row_ids`,
   then `self.pool.submit(self.host.gather, rows)`, and keep the Future. Don't wait.
2. `PrefetchedMemory.forward(cids)`: get the result (calling `start` first if
   nothing is pending), clear it, and flatten the heads, the same output as
   `NgramMemory.forward`.
3. `prefetch_all(model, ids)`: call `start` on every Engram's memory. Each
   Engram has its own `lookup` for compressing ids.

`enable_prefetch(model)` swaps the memories in for you.

## The tests check

- Same logits as the normal lookup, with or without `prefetch_all`
- Every fetch starts before the first layer runs
- `prefetch_all` returns immediately

`make see N=9` times an 8-layer model with two Engram layers and an 80 ms
simulated fetch each, so you can see the overlap.

## Going further

Paper sections 2.5 and 6.4. Placement is a trade-off: a deeper Engram layer
gives the fetch more compute to hide behind, but early layers help modeling
most. Section 2.5 also sketches a cache hierarchy: N-gram use is Zipfian, so hot
rows can sit in GPU memory and the long tail on SSD.
