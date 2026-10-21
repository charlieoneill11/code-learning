import torch

from engram.compress import build_lookup
from engram.data import World
from engram.engram import Engram, EngramConfig

WORLD = World()


def lookup():
    return build_lookup(WORLD.vocab)[0]  # your step 1 code


def make(d=32):
    torch.manual_seed(0)
    eng = Engram(EngramConfig(d_model=d, min_rows=101, head_dim=8), lookup())
    with torch.no_grad():  # wake the conv up so the tests can see it
        eng.conv.conv.weight.normal_()
    return eng


def test_shape():
    out = make()(torch.randn(2, 12, 32), torch.randint(0, WORLD.vocab_size, (2, 12)))
    assert out.shape == (2, 12, 32)


def test_stores_alpha():
    eng = make()
    eng(torch.randn(2, 12, 32), torch.randint(0, WORLD.vocab_size, (2, 12)))
    assert eng.last_alpha is not None and eng.last_alpha.shape == (2, 12, 1)
    assert not eng.last_alpha.requires_grad


def test_causal():
    eng = make()
    ids, h = torch.randint(0, WORLD.vocab_size, (2, 20)), torch.randn(2, 20, 32)
    ids2, h2 = ids.clone(), h.clone()
    ids2[:, 12:] = torch.randint(0, WORLD.vocab_size, (2, 8))
    h2[:, 12:] = torch.randn(2, 8, 32)
    assert torch.allclose(eng(h, ids)[:, :12], eng(h2, ids2)[:, :12], atol=1e-6)


def test_every_spelling_reads_the_same_memory():
    # The same name in lower, Title and UPPER case: different token IDs, same output.
    eng = make()
    name = WORLD.names[0]
    h = torch.randn(1, len(name), 32)
    outs = [eng(h, torch.tensor([[WORLD.name_ids[w][c] for w in name]])) for c in range(3)]
    assert torch.allclose(outs[0], outs[1]) and torch.allclose(outs[0], outs[2])


def test_includes_the_conv_branch():
    eng = make()
    ids, h = torch.randint(0, WORLD.vocab_size, (1, 15)), torch.randn(1, 15, 32)
    with_conv = eng(h, ids)
    with torch.no_grad():
        eng.conv.conv.weight.zero_()
    assert not torch.allclose(with_conv, eng(h, ids))


def test_matches_reference(fixture):
    f = fixture("07_engram")
    torch.manual_seed(0)
    eng = Engram(EngramConfig(d_model=32, min_rows=101, head_dim=8), lookup())
    eng.load_state_dict(f["state"])
    out = eng(f["h"], f["ids"])
    assert torch.allclose(eng.last_alpha, f["alpha"], atol=1e-5)
    assert torch.allclose(out, f["out"], atol=1e-5)
