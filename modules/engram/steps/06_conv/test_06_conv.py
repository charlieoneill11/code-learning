import torch
import torch.nn.functional as F

from engram.conv import ShortConv


def random_conv(dim=8):
    conv = ShortConv(dim)
    with torch.no_grad():
        conv.conv.weight.normal_()
    return conv


def test_shape():
    assert ShortConv(8)(torch.randn(2, 20, 8)).shape == (2, 20, 8)


def test_outputs_zero_at_init():
    assert torch.equal(ShortConv(8)(torch.randn(2, 20, 8)), torch.zeros(2, 20, 8))


def test_causal():
    conv = random_conv()
    x = torch.randn(2, 20, 8)
    later = x.clone()
    later[:, 10:] = torch.randn(2, 10, 8)
    assert torch.allclose(conv(x)[:, :10], conv(later)[:, :10])


def test_taps_are_t_t3_t6_t9():
    # Turn on one kernel weight at a time: weight j should read input t - 3 * (3 - j).
    conv = ShortConv(4)
    x = torch.randn(1, 20, 4)
    normed = conv.norm(x)
    for j, shift in [(3, 0), (2, 3), (1, 6), (0, 9)]:
        with torch.no_grad():
            conv.conv.weight.zero_()
            conv.conv.weight[:, 0, j] = 1.0
        expected = F.silu(F.pad(normed, (0, 0, shift, 0))[:, :20])
        assert torch.allclose(conv(x), expected, atol=1e-6), f"kernel weight {j} should read t-{shift}"


def test_depthwise_channels_do_not_mix():
    conv = random_conv(dim=4)
    with torch.no_grad():
        conv.norm.weight.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))  # only channel 0 survives the norm
    out = conv(torch.randn(1, 20, 4))
    assert out[..., 1:].abs().max() < 1e-6


def test_matches_reference(fixture):
    f = fixture("06_conv")
    conv = ShortConv(16)
    conv.load_state_dict(f["state"])
    assert torch.allclose(conv(f["x"]), f["out"], atol=1e-5)
