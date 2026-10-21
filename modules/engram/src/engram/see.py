"""See what your code does on the explainer's toy example.

    uv run python -m engram.see 3      # or: make see N=3

Each demo calls YOUR functions and prints what came out, so you can poke at a
step before (or instead of) reading test failures.
"""

import sys

import torch

the, great, of, wales, alexander = 2, 11, 4, 27, 7  # toy canonical IDs from the explainer


def step1() -> None:
    from .compress import build_lookup, normalize

    for text in [" Apple", "APPLE", "apple", "École", "  New\n\nYork ", "\n\n"]:
        print(f"  normalize({text!r:16}) -> {normalize(text)!r}")
    vocab = ["<pad>", " Apple", "apple", " APPLE", " pear", "Pear", " école", "ecole"]
    lookup, n = build_lookup(vocab)
    print(f"\n  vocab   {vocab}\n  lookup  {lookup.tolist()}  ({len(vocab)} tokens -> {n} canonical IDs)")
    from .data import World

    w = World()
    print(f"  the step 8 world: {w.vocab_size} tokens -> {build_lookup(w.vocab)[1]} canonical IDs")


def step2() -> None:
    from .ngrams import suffix_ngrams

    words = ["Only", "Alexander", "the", "Great"]
    ids = torch.tensor([[5, alexander, the, great]])
    for n in (2, 3):
        grams = suffix_ngrams(ids, n, pad_id=0)[0]
        print(f"  {n}-grams:")
        for t, g in enumerate(grams.tolist()):
            print(f"    t={t} {words[t]:<10} {g}")


def step3() -> None:
    from .hashing import ngram_hash

    m = torch.tensor([3, 5])
    for name, gram in [("(the, Great)", [the, great]), ("(Great, the)", [great, the]), ("(of, Wales)", [of, wales])]:
        row = ngram_hash(torch.tensor([gram]), m, 13).item()
        print(f"  {name:<13} -> row {row} of 13")
    rows = ngram_hash(torch.randint(0, 50_000, (200_000, 2)), torch.tensor([1_000_003, 999_983]), 1009)
    counts = torch.bincount(rows, minlength=1009)
    print(f"\n  200k random bigrams into 1009 rows: min {counts.min()}, mean {counts.float().mean():.0f}, max {counts.max()}")


def step4() -> None:
    from .memory import NgramMemory

    mem = NgramMemory(100, orders=(2,), heads=3, head_dim=4, table_sizes=[[13, 17, 19]], multipliers=torch.tensor([3, 5]))
    for name, gram in [("(the, Great)", [the, great]), ("(of, Wales)", [of, wales])]:
        rows = mem.row_ids(torch.tensor([gram]))[0, -1]
        local = (rows - mem.offsets).tolist()
        print(f"  {name:<13} rows per head {local}  global rows {rows.tolist()}")
    print(f"  memory vector e_t shape: {tuple(mem(torch.tensor([[the, great]])).shape)}")


def step5() -> None:
    from .gate import ContextGate

    torch.manual_seed(0)
    d = 16
    g = ContextGate(d, d)
    with torch.no_grad():
        g.W_k.weight.copy_(torch.eye(d))
        g.W_k.bias.zero_()
    h = torch.randn(1, 1, d)
    for name, e in [("agrees", h), ("unrelated", torch.randn(1, 1, d)), ("disagrees", -h)]:
        _, alpha = g(h, e)
        print(f"  memory {name:<10} alpha = {alpha.item():.3f}")


def step6() -> None:
    from .conv import ShortConv

    conv = ShortConv(8)
    x = torch.randn(1, 13, 8)
    print(f"  at init, output = {conv(x).abs().max().item():.1f} everywhere (zero-initialized)")
    with torch.no_grad():
        conv.conv.weight.fill_(1.0)
    base = conv(x)[0, 12]
    print("  which inputs move output t=12? (nudge one input at a time)")
    for s in range(13):
        x2 = x.clone()
        x2[0, s] += torch.randn(8)
        moved = (conv(x2)[0, 12] - base).abs().max().item() > 1e-6
        print(f"    input t={s:<2} {'yes' if moved else '-'}")


def step7() -> None:
    from .compress import build_lookup
    from .engram import Engram, EngramConfig

    vocab = ["<pad>", " the", " Great", " great", " GREAT", " Alexander"]
    lookup, _ = build_lookup(vocab)
    torch.manual_seed(0)
    eng = Engram(EngramConfig(d_model=32, min_rows=101), lookup)
    h = torch.randn(1, 3, 32)
    for ids in ([5, 1, 2], [5, 1, 3], [5, 1, 4]):
        out = eng(h, torch.tensor([ids]))
        words = "".join(vocab[i] for i in ids)
        print(f"  {words!r:24} update at last token: {out[0, -1, :4].tolist()}")
    print("  (same numbers: every spelling of 'Great' reads the same memory rows)")


def step8() -> None:
    print("  Run the full experiment:  uv run python -m engram.experiment  (or --quick)")


def step9() -> None:
    import time

    from .model import ModelConfig, TinyLM
    from .prefetch import enable_prefetch, forward_with_prefetch

    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=500, d_model=256, n_layers=8, n_heads=8, d_ff=1024, engram_layers=(1, 5))
    model = TinyLM(cfg).eval()
    ids = torch.randint(0, 500, (16, 256))
    enable_prefetch(model, latency=0.08)
    with torch.no_grad():
        model(ids)  # warm up
        t0 = time.perf_counter()
        model(ids)
        t1 = time.perf_counter()
        forward_with_prefetch(model, ids)
        t2 = time.perf_counter()
    print(f"  8-layer model, Engram at layers 1 and 5, 80 ms simulated host fetch each")
    print(f"  fetch when the layer needs it: {1e3 * (t1 - t0):6.0f} ms")
    print(f"  prefetch before the forward:   {1e3 * (t2 - t1):6.0f} ms")


def main() -> None:
    steps = {str(i): f for i, f in enumerate([step1, step2, step3, step4, step5, step6, step7, step8, step9], 1)}
    if len(sys.argv) != 2 or sys.argv[1] not in steps:
        print(__doc__)
        sys.exit(1)
    try:
        steps[sys.argv[1]]()
    except NotImplementedError as e:
        print(f"  Not written yet: {e}")


if __name__ == "__main__":
    main()
