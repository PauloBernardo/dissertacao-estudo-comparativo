"""Top-k sparse attention.

Keeps only the top-k attention scores per query; zeroes the rest before
softmax normalisation. This produces exactly k non-zero weights per query,
giving a hard sparsity guarantee.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKAttention(nn.Module):
    """Multi-head attention where each query attends to at most k keys.

    Parameters
    ----------
    topk_ratio : float
        Fraction of keys to keep per query (0 < r ≤ 1).
        The actual k = max(1, round(topk_ratio * seq_len)).
    """

    def __init__(self, topk_ratio: float = 0.25) -> None:
        super().__init__()
        self.topk_ratio = topk_ratio

    def forward(
        self,
        scores: torch.Tensor,          # (B, H, Q, K)
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply top-k masking and return softmax attention weights.

        Parameters
        ----------
        scores : (batch, heads, queries, keys) — raw attention logits
        mask : optional boolean mask (True = attend)

        Returns
        -------
        weights : (batch, heads, queries, keys) — sparse attention weights
        """
        k = max(1, round(self.topk_ratio * scores.size(-1)))

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        # Zero out all but the top-k values per query
        topk_vals, _ = scores.topk(k, dim=-1)
        threshold = topk_vals[..., -1:].detach()          # (B, H, Q, 1)
        sparse_scores = scores.masked_fill(scores < threshold, float("-inf"))

        return F.softmax(sparse_scores, dim=-1)
