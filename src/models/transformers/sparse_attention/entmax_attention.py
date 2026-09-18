"""α-Entmax attention (Peters et al., 2019).

Uses entmax-α instead of softmax. At α=1 it reduces to softmax; at α=2
to sparsemax. Values of 1 < α < 2 give intermediate sparsity.
For α=1.5 (entmax-1.5), this produces naturally sparse distributions.

Reference:
    Peters B. et al., "Sparse Sequence-to-Sequence Models", ACL 2019.
    Correia G. et al., "Adaptively Sparse Transformers", EMNLP 2019.

Histórico de correção (auditoria 2026-09-18)
--------------------------------------------
A versão anterior deste módulo usava o intervalo de bisseção
τ ∈ [(α−1)·z_min − 1, (α−1)·z_max − 1]. Nesse intervalo Σp ≥ 1 em toda parte,
de modo que a bisseção convergia sempre ao extremo superior e a soma dos pesos
antes da renormalização ficava entre ≈1,4 e ≈7 (medido). O resultado era uma
distribuição "recortada e renormalizada" — suporte {i : z_i > z_max − 1/(α−1)} —
e NÃO o entmax-α. O intervalo correto (Peters et al., 2019, Alg. 1) é
τ ∈ [(α−1)·z_max − 1, (α−1)·z_max − D^{1−α}], adotado abaixo. O backward usa a
Jacobiana exata do entmax (como no pacote `entmax`), em vez de diferenciar
através das iterações da bisseção.
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


class _EntmaxBisectFunction(torch.autograd.Function):
    """entmax-α por bisseção em τ (forward) com Jacobiana exata (backward).

    Forward:  p_i = [(α−1) z_i − τ]_+^{1/(α−1)},  τ tal que Σ p_i = 1.
    Backward: com s_i = p_i^{2−α} (zero fora do suporte),
              dz = s ⊙ dp − s · (Σ s ⊙ dp) / (Σ s).
    """

    @staticmethod
    def forward(ctx, z: torch.Tensor, alpha: float, n_iter: int) -> torch.Tensor:
        am1 = alpha - 1.0
        D = z.size(-1)

        # −inf (posições mascaradas) → peso zero; substitui por valor finito
        inf_mask = torch.isinf(z) & (z < 0)
        z_fin = z.masked_fill(inf_mask, -1e9)

        zs = am1 * z_fin                                   # (α−1)·z
        z_max = zs.max(dim=-1, keepdim=True).values
        # Intervalo que contém a raiz (Peters et al., 2019):
        #   τ_lo: p_max = 1 (todo o peso num único índice)
        #   τ_hi: distribuição uniforme sobre D índices
        tau_lo = z_max - 1.0
        tau_hi = z_max - float(D) ** (-am1)

        for _ in range(n_iter):
            tau_mid = (tau_lo + tau_hi) / 2.0
            p = (zs - tau_mid).clamp(min=0.0) ** (1.0 / am1)
            s = p.sum(dim=-1, keepdim=True)
            gt = s > 1.0            # soma grande demais → aumentar τ
            tau_lo = torch.where(gt, tau_mid, tau_lo)
            tau_hi = torch.where(gt, tau_hi, tau_mid)

        tau = (tau_lo + tau_hi) / 2.0
        p = (zs - tau).clamp(min=0.0) ** (1.0 / am1)
        p = p.masked_fill(inf_mask, 0.0)
        # Após n_iter bisseções |Σp − 1| ≈ 2^{-n_iter}; a renormalização é apenas
        # segurança numérica, não corrige um intervalo errado.
        p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        ctx.save_for_backward(p)
        ctx.alpha = alpha
        return p

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (p,) = ctx.saved_tensors
        alpha = ctx.alpha
        s = torch.where(p > 0, p ** (2.0 - alpha), torch.zeros_like(p))
        dz = grad_out * s
        q = dz.sum(dim=-1, keepdim=True) / s.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        dz = dz - q * s
        return dz, None, None


def _entmax_bisect(
    z: torch.Tensor,
    alpha: float,
    n_iter: int = 50,
    dim: int = -1,
) -> torch.Tensor:
    """Compute entmax-α along ``dim`` via bisection on the dual variable τ.

    p_i = max(0, (α-1) z_i - τ)^{1/(α-1)},  with τ such that Σ p_i = 1.
    Handles the general 1 < α ≤ 2 case; α = 2 coincides with sparsemax.
    """
    if dim != -1 and dim != z.dim() - 1:
        z = z.transpose(dim, -1)
        p = _EntmaxBisectFunction.apply(z.contiguous(), alpha, n_iter)
        return p.transpose(dim, -1)
    return _EntmaxBisectFunction.apply(z.contiguous(), alpha, n_iter)
