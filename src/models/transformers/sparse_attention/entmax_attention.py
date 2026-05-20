"""α-Entmax attention (Peters et al., 2019).

Uses entmax-α instead of softmax. At α=1 it reduces to softmax; at α=2
to sparsemax. Values of 1 < α < 2 give intermediate sparsity.
For α=1.5 (entmax-1.5), this produces naturally sparse distributions.

Reference:
    Peters B. et al., "Sparse Sequence-to-Sequence Models", ACL 2019.
    Correia G. et al., "Adaptively Sparse Transformers", EMNLP 2019.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EntmaxAttention(nn.Module):
    """Multi-head attention using α-entmax.

    Parameters
    ----------
    alpha : float
        Entmax parameter. 1.0 = softmax, 1.5 = entmax-1.5 (default),
        2.0 = sparsemax. 1 < alpha ≤ 2 guaranteed sparse.
    n_iter : int
        Bisection iterations for the entmax projection (default 50).
    """

    def __init__(self, alpha: float = 1.5, n_iter: int = 50) -> None:
        super().__init__()
        self.alpha = alpha
        self.n_iter = n_iter

    def forward(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        if abs(self.alpha - 1.0) < 1e-6:
            return F.softmax(scores, dim=-1)
        return _entmax_bisect(scores, self.alpha, self.n_iter, dim=-1)


def _entmax_bisect(
    z: torch.Tensor,
    alpha: float,
    n_iter: int = 50,
    dim: int = -1,
) -> torch.Tensor:
    """Compute entmax-α via bisection on the dual variable τ.

    p_i = max(0, ((α-1) z_i - τ))^{1/(α-1)}  (unnormalised)
    Find τ such that Σ p_i = 1.

    This implementation handles the general 1 < α ≤ 2 case.
    """
    # Move target dim to last
    z = z.transpose(dim, -1).contiguous()
    orig_shape = z.shape
    z = z.view(-1, z.size(-1))    # (N, D)
    N, D = z.shape

    am1 = alpha - 1.0             # α - 1

    # Replace -inf with -1e9 for numerical stability
    z_fin = z.clone()
    inf_mask = z == float("-inf")
    z_fin[inf_mask] = -1e9

    # Bracket the root via τ in [τ_lo, τ_hi]
    # τ_hi: when τ = (α-1) z_max, all but one weight is 0
    z_max = z_fin.max(dim=-1, keepdim=True).values
    z_min = z_fin.min(dim=-1, keepdim=True).values

    tau_hi = (am1 * z_max - 1.0)
    tau_lo = (am1 * z_min - 1.0)

    for _ in range(n_iter):
        tau_mid = (tau_hi + tau_lo) / 2.0
        p_unnorm = ((am1 * z_fin - tau_mid)).clamp(min=0.0)
        p = p_unnorm ** (1.0 / am1)
        p_sum = p.sum(dim=-1, keepdim=True)

        # Adjust bracket
        lo_mask = (p_sum < 1.0)
        tau_hi = torch.where(lo_mask, tau_mid, tau_hi)
        tau_lo = torch.where(lo_mask, tau_lo, tau_mid)

    # Final projection
    p_unnorm = (am1 * z_fin - tau_mid).clamp(min=0.0)
    p = p_unnorm ** (1.0 / am1)
    # Zero out originally masked positions
    p[inf_mask] = 0.0
    # Renormalise for numerical safety
    p_sum = p.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    p = p / p_sum

    return p.view(*orig_shape).transpose(dim, -1)
