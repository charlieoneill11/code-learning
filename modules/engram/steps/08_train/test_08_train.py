import torch
from torch import nn

from engram.compress import build_lookup
from engram.data import World, WorldConfig
from engram.model import Block, ModelConfig, TinyLM
from engram.train import TrainConfig, evaluate, train


class FakeEngram(nn.Module):
    """Adds 5 everywhere and remembers what it was called with."""

    def forward(self, h, ids):
        self.seen_h, self.seen_ids = h.clone(), ids
        return torch.full_like(h, 5.0)


def test_block_without_engram_is_two_residual_updates():
    torch.manual_seed(0)
    block = Block(ModelConfig(vocab_size=10, d_model=16, n_heads=2, d_ff=32))
    h = torch.randn(2, 6, 16)
    h1 = h + block.attn(block.norm1(h))
    expected = h1 + block.mlp(block.norm2(h1))
    assert torch.allclose(block(h, torch.zeros(2, 6, dtype=torch.long)), expected, atol=1e-6)


def test_engram_update_comes_first():
    torch.manual_seed(0)
    fake = FakeEngram()
    block = Block(ModelConfig(vocab_size=10, d_model=16, n_heads=2, d_ff=32), engram=fake)
    h, ids = torch.randn(2, 6, 16), torch.randint(0, 10, (2, 6))
    out = block(h, ids)
    h0 = h + 5.0
    h1 = h0 + block.attn(block.norm1(h0))
    assert torch.allclose(out, h1 + block.mlp(block.norm2(h1)), atol=1e-5)
    assert torch.equal(fake.seen_h, h) and torch.equal(fake.seen_ids, ids)


def test_tiny_model_with_engram_learns_the_facts():
    # About 15 seconds: a small world, 200 steps. Engram should nail most facts.
    world = World(WorldConfig(n_entities=300))
    lookup, _ = build_lookup(world.vocab)
    torch.manual_seed(0)
    model = TinyLM(ModelConfig(vocab_size=world.vocab_size, engram_layers=(1,)), lookup)
    train(model, world, TrainConfig(steps=200, eval_every=200), log=lambda *_: None)
    assert evaluate(model, world)["fact_acc"] > 0.8
