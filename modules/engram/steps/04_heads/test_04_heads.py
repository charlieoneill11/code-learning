import torch

from engram.memory import NgramMemory


def t(x):
    return torch.tensor(x, dtype=torch.long)


def toy(orders=(2,), table_sizes=([13, 17, 19],), multipliers=(3, 5), head_dim=4):
    return NgramMemory(
        vocab_size=100,
        orders=orders,
        head_dim=head_dim,
        table_sizes=[list(s) for s in table_sizes],
        multipliers=t(list(multipliers)),
    )


def test_toy_rows_per_head():
    # 43 mod 13 = 4, 43 mod 17 = 9, 43 mod 19 = 5 ...
    mem = toy()
    local = mem.row_ids(t([[2, 11]]))[0, 1] - mem.offsets
    assert local.tolist() == [4, 9, 5]


def test_toy_global_rows_use_offsets():
    # ... and head j's block starts at offsets [0, 13, 30]
    assert toy().row_ids(t([[2, 11]]))[0, 1].tolist() == [4, 22, 35]


def test_collision_in_one_head_only():
    # (of, Wales) shares row 4 with (the, Great) in head 0, but not in heads 1 and 2.
    mem = toy()
    great, wales = mem.row_ids(t([[2, 11]]))[0, 1], mem.row_ids(t([[4, 27]]))[0, 1]
    assert (great == wales).tolist() == [True, False, False]


def test_columns_are_orders_then_heads():
    mem = toy(orders=(2, 3), table_sizes=([13, 17], [19, 23]), multipliers=(3, 5, 7))
    rows = mem.row_ids(t([[7, 2, 11]]))[0, 2]
    # 2-gram (2, 11) mixes to 43; 3-gram (7, 2, 11) mixes to 26
    assert (rows - mem.offsets).tolist() == [43 % 13, 43 % 17, 26 % 19, 26 % 23]
    assert mem.offsets.tolist() == [0, 13, 30, 49]


def test_forward_concatenates_rows():
    mem = toy()
    cids = torch.randint(0, 100, (2, 6))
    e = mem(cids)
    assert e.shape == (2, 6, 3 * 4)
    rows = mem.row_ids(cids)
    for j in range(3):
        assert torch.equal(e[..., 4 * j : 4 * (j + 1)], mem.table.weight[rows[..., j]])


def test_gradients_reach_the_table():
    mem = toy()
    mem(t([[2, 11, 4, 27]])).sum().backward()
    assert mem.table.weight.grad.abs().sum() > 0


def test_matches_reference(fixture):
    f = fixture("04_memory")
    torch.manual_seed(0)
    mem = NgramMemory(vocab_size=5000, orders=(2, 3), heads=4, min_rows=211, head_dim=8, seed=3)
    mem.load_state_dict(f["state"])
    assert torch.equal(mem.row_ids(f["cids"]), f["row_ids"])
    assert torch.allclose(mem(f["cids"]), f["e"])


def test_more_heads_means_fewer_full_collisions():
    # How many pairs of different bigrams collide in EVERY head? One head vs four.
    torch.manual_seed(0)
    grams = torch.unique(torch.randint(1, 3000, (4000, 2)), dim=0)
    cids = grams.view(-1, 1, 2).flatten(1)  # each bigram as its own 2-token sequence

    def full_collisions(heads):
        mem = NgramMemory(vocab_size=3000, orders=(2,), heads=heads, min_rows=211, seed=1)
        rows = mem.row_ids(cids)[:, 1]  # rows at the second token = the bigram
        return len(rows) - len(torch.unique(rows, dim=0))

    one, four = full_collisions(1), full_collisions(4)
    assert one > 100
    assert four < one / 20
