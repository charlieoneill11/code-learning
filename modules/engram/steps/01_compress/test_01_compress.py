import torch

from engram.compress import build_lookup, normalize


def test_case_and_leading_space():
    for text in [" Apple", "APPLE", "apple", "  aPPle  "]:
        assert normalize(text) == "apple"


def test_accents_are_dropped():
    assert normalize("École") == "ecole"
    assert normalize("naïve") == "naive"


def test_nfkc_folds_lookalike_characters():
    assert normalize("ﬁne") == "fine"  # the "fi" ligature
    assert normalize("ＡＢＣ") == "abc"  # full-width letters


def test_whitespace_runs_become_one_space():
    assert normalize("New \t\n York") == "new york"


def test_whitespace_only_token_becomes_one_space():
    for text in [" ", "\n\n", "\t", "  \r\n "]:
        assert normalize(text) == " "


def test_empty_stays_empty():
    assert normalize("") == ""


def test_lookup_merges_spellings():
    vocab = ["<pad>", " Apple", "apple", "APPLE", " pear", "Pear"]
    lookup, n = build_lookup(vocab)
    assert lookup.tolist() == [0, 1, 1, 1, 2, 2]
    assert n == 3
    assert lookup.dtype == torch.long


def test_ids_follow_first_appearance():
    lookup, n = build_lookup(["b", "A", "B", "a", "c"])
    assert lookup.tolist() == [0, 1, 0, 1, 2]
    assert n == 3


def test_empty_normalization_keeps_raw_text():
    # A lone combining accent normalizes to "", so each keeps its own raw text as the key.
    lookup, n = build_lookup(["a", "́", "̀", "́"])
    assert lookup.tolist() == [0, 1, 2, 1]
    assert n == 3


def test_step8_world_shrinks():
    from engram.data import World

    world = World()
    lookup, n = build_lookup(world.vocab)
    assert world.vocab_size == 282 and n == 202
    assert int(lookup.max()) == n - 1
