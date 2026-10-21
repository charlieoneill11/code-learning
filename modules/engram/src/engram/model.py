"""A tiny decoder-only transformer, with optional Engram layers.

Step 8 is wiring Engram into Block.forward. Everything else here is plumbing.
"""

from dataclasses import dataclass, field, replace

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .engram import Engram, EngramConfig


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 256
    max_len: int = 256
    engram_layers: tuple[int, ...] = ()
    engram: EngramConfig = field(default_factory=EngramConfig)


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, D // self.n_heads).unbind(2)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, engram: Engram | None = None):
        super().__init__()
        self.engram = engram
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn = Attention(cfg.d_model, cfg.n_heads)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff)

    def forward(self, h: Tensor, ids: Tensor) -> Tensor:
        """h: (B, T, d_model) hidden state. ids: (B, T) raw token IDs.

        A standard pre-norm block is two residual updates:
            h = h + attn(norm1(h))
            h = h + mlp(norm2(h))
        If this block has an Engram (self.engram is not None), add its update
        to h first, before attention. Engram does its own normalization, so
        pass it h as is.
        """
        # >>> step 8: your code
        raise NotImplementedError("step 8")
        # <<<


class TinyLM(nn.Module):
    def __init__(self, cfg: ModelConfig, lookup: Tensor | None = None, pad_id: int = 0):
        """lookup: token ID -> canonical ID table (step 1). Needed only when
        cfg.engram_layers is non-empty; None means no compression."""
        super().__init__()
        self.cfg = cfg
        if lookup is None:
            lookup = torch.arange(cfg.vocab_size)
        taken: set[int] = set()  # keeps every layer's primes distinct
        engram_cfg = replace(cfg.engram, d_model=cfg.d_model)
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            Block(
                cfg,
                Engram(engram_cfg, lookup, pad_id, layer_id=i, taken_primes=taken)
                if i in cfg.engram_layers
                else None,
            )
            for i in range(cfg.n_layers)
        )
        self.norm = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(_init)
        for e in self.engrams():
            nn.init.zeros_(e.conv.conv.weight)

    def forward(self, ids: Tensor) -> Tensor:
        """ids: (B, T) -> logits (B, T, vocab_size)."""
        T = ids.shape[1]
        h = self.tok(ids) + self.pos(torch.arange(T, device=ids.device))
        for block in self.blocks:
            h = block(h, ids)
        return self.head(self.norm(h))

    def engrams(self) -> list[Engram]:
        return [b.engram for b in self.blocks if b.engram is not None]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _init(m: nn.Module) -> None:
    if isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.normal_(m.weight, std=0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
