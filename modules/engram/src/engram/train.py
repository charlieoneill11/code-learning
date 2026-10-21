"""Training and evaluation loop for the step 8 experiment. Plumbing only."""

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .data import World
from .model import TinyLM


@dataclass
class TrainConfig:
    steps: int = 1500
    batch_size: int = 32
    lr: float = 3e-3
    warmup: int = 100
    eval_every: int = 250
    eval_batches: int = 8
    seed: int = 0


def evaluate(model: TinyLM, world: World, n_batches: int = 8, seed: int = 12345) -> dict:
    """Loss over all tokens, plus loss and accuracy on fact tokens only."""
    rng = np.random.default_rng(seed)
    model.eval()
    tot_loss = tot_n = fact_loss = fact_hits = fact_n = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            ids, is_fact = world.sample(32, rng)
            x, y, m = ids[:, :-1], ids[:, 1:], is_fact[:, 1:]
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="none").view_as(y)
            tot_loss += loss.sum().item()
            tot_n += y.numel()
            fact_loss += loss[m].sum().item()
            fact_hits += (logits.argmax(-1) == y)[m].sum().item()
            fact_n += m.sum().item()
    model.train()
    return {
        "loss": tot_loss / tot_n,
        "fact_loss": fact_loss / fact_n,
        "fact_acc": fact_hits / fact_n,
    }


def train(model: TinyLM, world: World, cfg: TrainConfig = TrainConfig(), log=print) -> list[dict]:
    """Train with AdamW and a warmup + cosine schedule. Returns eval history."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0)
    history = []
    start = time.time()
    for step in range(1, cfg.steps + 1):
        lr = cfg.lr * min(1.0, step / cfg.warmup)
        lr *= 0.5 * (1 + math.cos(math.pi * min(1.0, step / cfg.steps)))
        for group in opt.param_groups:
            group["lr"] = lr
        ids, _ = world.sample(cfg.batch_size, rng)
        logits = model(ids[:, :-1])
        loss = F.cross_entropy(logits.flatten(0, 1), ids[:, 1:].flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            metrics = {"step": step, "secs": time.time() - start, **evaluate(model, world, cfg.eval_batches)}
            history.append(metrics)
            log(
                f"  step {step:5d}  loss {metrics['loss']:.3f}  "
                f"fact acc {metrics['fact_acc']:6.1%}  ({metrics['secs']:.0f}s)"
            )
    return history
