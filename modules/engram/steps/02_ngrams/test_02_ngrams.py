import torch

from engram.ngrams import suffix_ngrams


def t(x):
    return torch.tensor(x, dtype=torch.long)


def test_docstring_example():
    out = suffix_ngrams(t([[5, 6, 7]]), n=3, pad_id=0)
    assert out.tolist() == [[[0, 0, 5], [0, 5, 6], [5, 6, 7]]]


def test_alexander_the_great():
    # Only=5 Alexander=7 the=2 Great=11: at "Great" the suffixes are (the, Great) and (Alexander, the, Great).
    ids = t([[5, 7, 2, 11]])
    assert suffix_ngrams(ids, 2, pad_id=0)[0, 3].tolist() == [2, 11]
    assert suffix_ngrams(ids, 3, pad_id=0)[0, 3].tolist() == [7, 2, 11]


def test_pad_id_is_used():
    assert suffix_ngrams(t([[4, 4]]), 3, pad_id=9).tolist() == [[[9, 9, 4], [9, 4, 4]]]


def test_n_equals_1_is_just_the_tokens():
    ids = torch.randint(0, 100, (2, 7))
    assert torch.equal(suffix_ngrams(ids, 1, pad_id=0), ids.unsqueeze(-1))


def test_shape_and_dtype():
    out = suffix_ngrams(torch.randint(0, 100, (4, 10)), 3, pad_id=0)
    assert out.shape == (4, 10, 3) and out.dtype == torch.long


def test_rows_in_a_batch_are_independent():
    a, b = torch.randint(1, 100, (1, 8)), torch.randint(1, 100, (1, 8))
    both = suffix_ngrams(torch.cat([a, b]), 3, pad_id=0)
    assert torch.equal(both[1:], suffix_ngrams(b, 3, pad_id=0))


def test_never_reads_ahead():
    ids = torch.randint(1, 100, (2, 12))
    later = ids.clone()
    later[:, 6:] = torch.randint(1, 100, (2, 6))
    assert torch.equal(suffix_ngrams(ids, 3, 0)[:, :6], suffix_ngrams(later, 3, 0)[:, :6])
