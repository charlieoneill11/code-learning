"""Step 6: the short convolution.

Each position's memory is fetched on its own. A small causal convolution
lets neighbouring memories mix: position t sees the gated values at
t, t-3, t-6 and t-9 (kernel 4, dilation 3 = the longest N-gram order).

It is depthwise (each channel has its own 4 weights, channels never mix) and
its weights start at zero, so at initialization the branch outputs exactly 0
and Engram behaves as if the convolution weren't there.
"""

from torch import Tensor, nn
import torch.nn.functional as F


class ShortConv(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 4, dilation: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.norm = nn.RMSNorm(dim)
        # groups=dim makes it depthwise. No padding here: making it causal is your job.
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, dilation=dilation, bias=False)
        nn.init.zeros_(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, dim) -> (B, T, dim) equal to SiLU(causal_conv(RMSNorm(x))).

        Causal: output t may only depend on inputs at t and earlier. Pad on the
        left only, and remember nn.Conv1d wants (B, channels, T).
        """
        # >>> step 6: your code
        y = self.norm(x).transpose(1, 2)
        y = F.pad(y, ((self.kernel_size - 1) * self.dilation, 0))
        y = self.conv(y)
        return F.silu(y).transpose(1, 2)
        # <<<
