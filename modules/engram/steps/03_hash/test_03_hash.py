import torch

from engram.hashing import ngram_hash


def t(x):
    return torch.tensor(x, dtype=torch.long)


def test_toy_example_from_explainer():
    # (the, Great) = (2, 11): 11*3 XOR 2*5 = 33 XOR 10 = 43, and 43 mod 13 = 4
    assert ngram_hash(t([[2, 11]]), t([3, 5]), 13).tolist() == [4]


def test_order_matters():
    # (Great, the): 2*3 XOR 11*5 = 6 XOR 55 = 49, and 49 mod 13 = 10
    assert ngram_hash(t([[11, 2]]), t([3, 5]), 13).tolist() == [10]


def test_collision_from_explainer():
    # (of, Wales) = (4, 27): 27*3 XOR 4*5 = 81 XOR 20 = 69, and 69 mod 13 = 4. Same row as (the, Great)!
    assert ngram_hash(t([[4, 27]]), t([3, 5]), 13).tolist() == [4]


def test_trigram():
    # newest*3 XOR middle*5 XOR oldest*7 = 33 XOR 10 XOR 49 = 26, and 26 mod 13 = 0
    assert ngram_hash(t([[7, 2, 11]]), t([3, 5, 7]), 13).tolist() == [0]


def test_works_on_any_leading_shape():
    grams = torch.randint(0, 1000, (2, 5, 3))
    out = ngram_hash(grams, t([3, 5, 7]), 101)
    assert out.shape == (2, 5)
    assert out.dtype == torch.long
    assert int(out.min()) >= 0 and int(out.max()) < 101


def test_matches_reference(fixture):
    f = fixture("03_hash")
    assert torch.equal(ngram_hash(f["ngrams"], f["multipliers"], f["table_size"]), f["expected3"])
    assert torch.equal(ngram_hash(f["ngrams"][..., 1:], f["multipliers"][:2], f["table_size"]), f["expected2"])


def test_rows_spread_evenly():
    torch.manual_seed(0)
    rows = ngram_hash(torch.randint(0, 50_000, (200_000, 2)), t([1_000_003, 999_983]), 1009)
    counts = torch.bincount(rows, minlength=1009).float()
    assert counts.max() < 2 * counts.mean()
