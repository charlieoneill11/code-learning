"""Step 2: suffix N-grams.

At every position t, Engram looks at the last few tokens ending at t: the
2-gram (x[t-1], x[t]) and the 3-gram (x[t-2], x[t-1], x[t]). It never looks
ahead, so the model stays causal. Before the start of the sequence there are
no tokens, so those slots are filled with a pad ID.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def suffix_ngrams(ids: Tensor, n: int, pad_id: int) -> Tensor:
    """Return the n-gram ending at every position.

    ids: (B, T) int64 token IDs.
    Returns (B, T, n) int64 where out[b, t] = (x[t-n+1], ..., x[t-1], x[t]),
    oldest first and newest last. Slots before the start hold pad_id.

    Example with n=3, pad_id=0 and ids [[5, 6, 7]]:
        [[[0, 0, 5],
          [0, 5, 6],
          [5, 6, 7]]]
    """
    # >>> step 2: your code
    T = ids.shape[1]
    padded = F.pad(ids, (n - 1, 0), value=pad_id)
    return torch.stack([padded[:, k : k + T] for k in range(n)], dim=-1)
    # <<<
