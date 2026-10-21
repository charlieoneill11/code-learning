"""Step 9 (stretch): prefetch rows from host memory.

In the paper the Engram table is enormous (up to 100B parameters), far too big
for GPU memory, so it lives in ordinary CPU RAM ("host memory"). Copying rows
across is slow, but there's a trick: the rows a layer will read depend only on
the token IDs, which are known before the forward pass starts. So you can work
out every layer's rows up front and start copying them while the earlier
layers compute. By the time the Engram layer runs, its rows have arrived.

Here the "GPU" is simply the model and the "host" is HostTable, which hands
back rows after a simulated transfer delay. Inference only: rows fetched this
way carry no gradient.
"""

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

import torch
from torch import Tensor, nn

from .compress import compress
from .memory import NgramMemory


class HostTable:
    """A copy of one Engram table held in "host memory".

    gather(rows) returns table[rows] after waiting latency + bytes / bandwidth
    seconds, to stand in for a host-to-device copy. Every call is recorded in
    self.events as ("fetch_start", name, time) and ("fetch_end", name, time).
    """

    def __init__(self, weight: Tensor, name: str = "", latency: float = 0.02, bytes_per_sec: float = 2e9):
        self.weight = weight.detach().clone()
        self.name = name
        self.latency = latency
        self.bytes_per_sec = bytes_per_sec
        self.events: list[tuple[str, str, float]] = []
        self._lock = threading.Lock()

    def gather(self, rows: Tensor) -> Tensor:
        self._log("fetch_start")
        out = self.weight[rows]
        time.sleep(self.latency + out.numel() * out.element_size() / self.bytes_per_sec)
        self._log("fetch_end")
        return out

    def _log(self, what: str) -> None:
        with self._lock:
            self.events.append((what, self.name, time.perf_counter()))


class PrefetchedMemory(nn.Module):
    """Drop-in replacement for NgramMemory whose rows come from a HostTable.

    Call start(cids) as early as you can; forward(cids) then picks up the
    result. If nobody called start, forward fetches on the spot (slow path).
    """

    def __init__(self, memory: NgramMemory, host: HostTable, pool: ThreadPoolExecutor):
        super().__init__()
        self.memory = memory  # still used for row_ids (hashing); its table is not read
        self.out_dim = memory.out_dim
        self.host = host
        self.pool = pool
        self._pending: Future | None = None

    def start(self, cids: Tensor) -> None:
        """Work out which rows these tokens need and begin fetching them in the
        background: submit self.host.gather to self.pool and keep the Future
        in self._pending. Don't wait for it."""
        # >>> step 9: your code
        raise NotImplementedError("step 9")
        # <<<

    def forward(self, cids: Tensor) -> Tensor:
        """Same output as NgramMemory.forward: (B, T, n_heads * head_dim).

        Wait for the pending fetch (calling start first if nothing is pending),
        clear self._pending, and flatten the heads together.
        """
        # >>> step 9: your code
        raise NotImplementedError("step 9")
        # <<<


def prefetch_all(model: nn.Module, ids: Tensor) -> None:
    """Start the fetch for every Engram layer in the model, before the forward
    pass. Each Engram has its own lookup table (engram.lookup) for turning
    ids into canonical IDs, and its memory is a PrefetchedMemory."""
    # >>> step 9: your code
    raise NotImplementedError("step 9")
    # <<<


def enable_prefetch(model: nn.Module, latency: float = 0.02, bytes_per_sec: float = 2e9) -> list[HostTable]:
    """Move every Engram table in the model to a HostTable and swap in a
    PrefetchedMemory. Returns the host tables (their .events are useful)."""
    pool = ThreadPoolExecutor(max_workers=max(1, len(model.engrams())))
    hosts = []
    for i, engram in enumerate(model.engrams()):
        host = HostTable(engram.memory.table.weight, f"engram{i}", latency, bytes_per_sec)
        engram.memory = PrefetchedMemory(engram.memory, host, pool)
        hosts.append(host)
    return hosts


@torch.no_grad()
def forward_with_prefetch(model: nn.Module, ids: Tensor) -> Tensor:
    prefetch_all(model, ids)
    return model(ids)
