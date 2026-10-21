"""Regenerate the golden test fixtures from the reference solution.

Run on the `solutions` branch (on main the stubs raise NotImplementedError):
    uv run python modules/engram/tools/make_fixtures.py
"""

from pathlib import Path

import torch

from engram.compress import build_lookup
from engram.conv import ShortConv
from engram.data import World
from engram.engram import Engram, EngramConfig
from engram.gate import ContextGate
from engram.hashing import make_multipliers, ngram_hash
from engram.memory import NgramMemory

OUT = Path(__file__).resolve().parents[1] / "fixtures"


def randomize(module: torch.nn.Module) -> None:
    """Give every parameter (norm weights and conv included) random values."""
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn_like(p) * 0.5 + (1.0 if p.dim() == 1 else 0.0))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    g = torch.Generator().manual_seed(0)

    # Step 3: hashing, realistic sizes.
    m = make_multipliers(3, 200_000, seed=1)
    grams = torch.randint(0, 200_000, (64, 50, 3), generator=g)
    torch.save(
        {
            "ngrams": grams,
            "multipliers": m,
            "table_size": 1_000_003,
            "expected3": ngram_hash(grams, m, 1_000_003),
            "expected2": ngram_hash(grams[..., 1:], m[:2], 1_000_003),
        },
        OUT / "03_hash.pt",
    )

    # Step 4: memory.
    torch.manual_seed(0)
    mem = NgramMemory(vocab_size=5000, orders=(2, 3), heads=4, min_rows=211, head_dim=8, seed=3)
    cids = torch.randint(0, 5000, (4, 40), generator=g)
    torch.save(
        {"state": mem.state_dict(), "cids": cids, "row_ids": mem.row_ids(cids), "e": mem(cids)},
        OUT / "04_memory.pt",
    )

    # Step 5: gate.
    torch.manual_seed(0)
    gate = ContextGate(48, 32)
    randomize(gate)
    h, e = torch.randn(3, 20, 32, generator=g), torch.randn(3, 20, 48, generator=g)
    out, alpha = gate(h, e)
    torch.save({"state": gate.state_dict(), "h": h, "e": e, "out": out, "alpha": alpha}, OUT / "05_gate.pt")

    # Step 6: short conv.
    torch.manual_seed(0)
    conv = ShortConv(16)
    randomize(conv)
    x = torch.randn(3, 30, 16, generator=g)
    torch.save({"state": conv.state_dict(), "x": x, "out": conv(x)}, OUT / "06_conv.pt")

    # Step 7: the whole Engram module.
    world = World()
    lookup, _ = build_lookup(world.vocab)
    torch.manual_seed(0)
    eng = Engram(EngramConfig(d_model=32, min_rows=101, head_dim=8), lookup)
    randomize(eng)
    ids = torch.randint(0, world.vocab_size, (3, 30), generator=g)
    h = torch.randn(3, 30, 32, generator=g)
    out = eng(h, ids)
    torch.save(
        {"state": eng.state_dict(), "ids": ids, "h": h, "out": out, "alpha": eng.last_alpha},
        OUT / "07_engram.pt",
    )
    print(f"wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
