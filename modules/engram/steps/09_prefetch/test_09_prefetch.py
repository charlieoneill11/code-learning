import time

import torch

from engram.compress import build_lookup
from engram.data import World
from engram.model import ModelConfig, TinyLM
from engram.prefetch import enable_prefetch, prefetch_all

WORLD = World()


def lookup():
    return build_lookup(WORLD.vocab)[0]  # your step 1 code


def model_and_ids():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=WORLD.vocab_size, n_layers=4, engram_layers=(1, 3))
    model = TinyLM(cfg, lookup()).eval()
    for e in model.engrams():
        with torch.no_grad():
            e.conv.conv.weight.normal_()
    return model, torch.randint(0, WORLD.vocab_size, (2, 24))


def test_same_output_as_normal_lookup():
    model, ids = model_and_ids()
    with torch.no_grad():
        expected = model(ids)
        enable_prefetch(model, latency=0.0)
        prefetch_all(model, ids)
        assert torch.allclose(model(ids), expected, atol=1e-5)


def test_works_without_prefetch_too():
    model, ids = model_and_ids()
    with torch.no_grad():
        expected = model(ids)
        enable_prefetch(model, latency=0.0)
        assert torch.allclose(model(ids), expected, atol=1e-5)  # nobody called prefetch_all


def test_fetches_start_before_the_first_layer_runs():
    model, ids = model_and_ids()
    hosts = enable_prefetch(model, latency=0.01)
    first_layer_at = []
    model.blocks[0].register_forward_pre_hook(lambda *_: first_layer_at.append(time.perf_counter()))
    with torch.no_grad():
        prefetch_all(model, ids)
        model(ids)
    starts = [when for host in hosts for what, _, when in host.events if what == "fetch_start"]
    assert len(starts) == len(hosts) == 2
    assert max(starts) < first_layer_at[0]


def test_prefetch_does_not_block():
    model, ids = model_and_ids()
    enable_prefetch(model, latency=0.3)
    t0 = time.perf_counter()
    prefetch_all(model, ids)
    assert time.perf_counter() - t0 < 0.1, "prefetch_all should start fetches, not wait for them"
    with torch.no_grad():
        model(ids)  # collect them so the threads finish
