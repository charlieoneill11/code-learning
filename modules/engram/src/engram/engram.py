"""Step 7: assemble the Engram module.

Everything from steps 1 to 6, in the order a token goes through it:

    token IDs -> canonical IDs -> N-grams -> hash heads -> rows -> e_t
    e_t + hidden state h_t -> gate -> alpha * v -> + short conv -> update

The module returns an update; the transformer block adds it to h (step 8).
"""

from dataclasses import dataclass

from torch import Tensor, nn

from .compress import compress
from .conv import ShortConv
from .gate import ContextGate
from .memory import NgramMemory


@dataclass
class EngramConfig:
    d_model: int = 64
    orders: tuple[int, ...] = (2, 3)
    heads: int = 4
    min_rows: int = 1009
    head_dim: int = 16
    kernel_size: int = 4
    seed: int = 0


class Engram(nn.Module):
    def __init__(
        self,
        cfg: EngramConfig,
        lookup: Tensor,
        pad_id: int = 0,
        layer_id: int = 0,
        taken_primes: set[int] | None = None,
    ):
        """lookup: token ID -> canonical ID table from step 1.
        layer_id: which transformer block this sits in (each layer gets its own
        multipliers and primes, like the paper)."""
        super().__init__()
        self.register_buffer("lookup", lookup.long(), persistent=False)
        n_canonical = int(lookup.max()) + 1
        self.memory = NgramMemory(
            vocab_size=n_canonical,
            orders=cfg.orders,
            heads=cfg.heads,
            min_rows=cfg.min_rows,
            head_dim=cfg.head_dim,
            pad_id=int(lookup[pad_id]),
            seed=cfg.seed + 10007 * layer_id,
            taken_primes=taken_primes,
        )
        self.gate = ContextGate(self.memory.out_dim, cfg.d_model)
        self.conv = ShortConv(cfg.d_model, cfg.kernel_size, dilation=max(cfg.orders))
        self.last_alpha: Tensor | None = None
        self.enabled = True  # experiment.py switches this off for the knockout test

    def forward(self, h: Tensor, ids: Tensor) -> Tensor:
        """h: (B, T, d_model) hidden state. ids: (B, T) raw token IDs.

        Returns the update to add to h, shape (B, T, d_model):
            v_gated + self.conv(v_gated)
        where v_gated is the gate's output. Also store the gate's alpha in
        self.last_alpha (detached), so experiments can look at it.
        """
        if not self.enabled:
            return h.new_zeros(h.shape)
        # >>> step 7: your code
        cids = compress(ids, self.lookup)
        e = self.memory(cids)
        v, alpha = self.gate(h, e)
        self.last_alpha = alpha.detach()
        return v + self.conv(v)
        # <<<
