"""Step 5: the context gate.

A row in the table is a blind guess: it was found by hashing, so it may belong
to some other N-gram that collided, or the phrase may mean something else
here. The model's hidden state h_t already knows the context. So Engram asks:
does this memory agree with what I'm currently thinking?

    k = W_k e        (what the memory claims)
    v = W_v e        (what the memory would add)
    alpha = sigmoid( RMSNorm(h) . RMSNorm(k) / sqrt(d) )
    output = alpha * v

Both sides are RMS-normalized, so the dot product is sqrt(d) times the cosine
of the angle between them: alpha is near 1 when they point the same way and
near 0 when they point opposite ways.
"""

import math

from torch import Tensor, nn


class ContextGate(nn.Module):
    def __init__(self, mem_dim: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.W_k = nn.Linear(mem_dim, d_model)
        self.W_v = nn.Linear(mem_dim, d_model)
        self.norm_h = nn.RMSNorm(d_model)
        self.norm_k = nn.RMSNorm(d_model)

    def forward(self, h: Tensor, e: Tensor) -> tuple[Tensor, Tensor]:
        """h: (B, T, d_model) hidden state. e: (B, T, mem_dim) memory vector.

        Returns (alpha * v, alpha) with shapes (B, T, d_model) and (B, T, 1).
        Use self.norm_h on h and self.norm_k on k.
        """
        # >>> step 5: your code
        raise NotImplementedError("step 5")
        # <<<
