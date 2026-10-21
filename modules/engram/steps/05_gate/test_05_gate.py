import math

import torch

from engram.gate import ContextGate


def identity_gate(d=16):
    """A gate whose key projection is the identity, so k = e."""
    g = ContextGate(d, d)
    with torch.no_grad():
        g.W_k.weight.copy_(torch.eye(d))
        g.W_k.bias.zero_()
    return g


def test_shapes():
    out, alpha = ContextGate(48, 32)(torch.randn(2, 5, 32), torch.randn(2, 5, 48))
    assert out.shape == (2, 5, 32) and alpha.shape == (2, 5, 1)


def test_agreeing_memory_opens_the_gate():
    d = 16
    h = torch.randn(1, 3, d)
    _, alpha = identity_gate(d)(h, 2.5 * h)  # same direction, different size
    assert torch.allclose(alpha, torch.full_like(alpha, 1 / (1 + math.exp(-math.sqrt(d)))), atol=1e-4)


def test_disagreeing_memory_closes_the_gate():
    d = 16
    h = torch.randn(1, 3, d)
    _, alpha = identity_gate(d)(h, -h)
    assert (alpha < 0.02).all()


def test_unrelated_memory_gives_one_half():
    h = torch.zeros(1, 1, 16)
    e = torch.zeros(1, 1, 16)
    h[..., 0], e[..., 1] = 1.0, 1.0  # orthogonal
    _, alpha = identity_gate()(h, e)
    assert torch.allclose(alpha, torch.tensor(0.5))


def test_output_is_alpha_times_value():
    g = ContextGate(48, 32)
    e = torch.randn(2, 5, 48)
    out, alpha = g(torch.randn(2, 5, 32), e)
    assert torch.allclose(out, alpha * g.W_v(e), atol=1e-6)


def test_loudness_of_h_does_not_matter():
    g = ContextGate(48, 32)
    h, e = torch.randn(2, 5, 32), torch.randn(2, 5, 48)
    assert torch.allclose(g(h, e)[1], g(100 * h, e)[1], atol=1e-5)


def test_gradients_flow_to_both_projections():
    g = ContextGate(48, 32)
    out, _ = g(torch.randn(2, 5, 32), torch.randn(2, 5, 48))
    out.sum().backward()
    assert g.W_k.weight.grad.abs().sum() > 0 and g.W_v.weight.grad.abs().sum() > 0


def test_matches_reference(fixture):
    f = fixture("05_gate")
    g = ContextGate(48, 32)
    g.load_state_dict(f["state"])
    out, alpha = g(f["h"], f["e"])
    assert torch.allclose(alpha, f["alpha"], atol=1e-5)
    assert torch.allclose(out, f["out"], atol=1e-5)
