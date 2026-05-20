"""Sparsemax attention (Martins & Astudillo, 2016).

Replaces softmax with sparsemax, which projects onto the probability
simplex, giving exact zeros for non-attended tokens.

Reference:
    Martins A. & Astudillo R., "From Softmax to Sparsemax: A Sparse Model
    of Attention and Multi-Label Classification", ICML 2016.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SparsemaxAttention(nn.Module):
    """Multi-head attention using sparsemax instead of softmax."""

    def forward(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        return _sparsemax(scores, dim=-1)


def _sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax projection onto the probability simplex.

    Algorithm: sort in descending order, find the support threshold,
    then clip and renormalise.

    Parameters
    ----------
    z : input tensor (any shape)
    dim : dimension along which to compute sparsemax

    Returns
    -------
    p : sparse probability distribution (same shape as z)
    """
    # Move target dimension to last for convenience
    z = z.transpose(dim, -1)
    orig_shape = z.shape
    z = z.contiguous().view(-1, z.size(-1))     # (N, D)

    # Sort descending
    z_sorted, _ = z.sort(dim=-1, descending=True)
    D = z.size(-1)
    k = torch.arange(1, D + 1, dtype=z.dtype, device=z.device)   # (D,)

    # Cumulative sum condition: z_sorted[j] - (cumsum - 1) / (j + 1) > 0
    z_cumsum = z_sorted.cumsum(dim=-1)
    support = z_sorted > (z_cumsum - 1.0) / k.unsqueeze(0)       # (N, D)
    k_z = support.sum(dim=-1, keepdim=True).float()               # (N, 1)
    tau = (z_cumsum.gather(dim=-1, index=(k_z.long() - 1).clamp(0, D - 1)) - 1.0) / k_z

    p = (z - tau).clamp(min=0.0)
    p = p.view(*orig_shape).transpose(dim, -1)
    return p
