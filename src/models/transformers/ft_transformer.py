"""FT-Transformer (Feature Tokenizer + Transformer) for tabular classification.

Supports four attention variants via the `attention_type` parameter:
    - "softmax"   : standard scaled dot-product attention (Gorishniy et al., 2021)
    - "topk"      : top-k sparse attention (hard sparsity)
    - "entmax"    : α-entmax attention (Peters et al., 2019)
    - "sparsemax" : sparsemax attention (Martins & Astudillo, 2016)

Architecture:
    - Numerical feature tokenizer: linear projection + bias per feature
    - [CLS] token prepended to the sequence
    - L Transformer encoder blocks (MultiHeadAttention + FFN + LayerNorm)
    - Classification head on the [CLS] token representation

References:
    Gorishniy Y. et al., "Revisiting Deep Learning Models for Tabular Data",
    NeurIPS 2021.
"""

from __future__ import annotations

import logging
import math
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .sparse_attention.entmax_attention import EntmaxAttention
from .sparse_attention.sparsemax_attention import SparsemaxAttention
from .sparse_attention.topk_attention import TopKAttention

logger = logging.getLogger(__name__)

AttentionType = Literal["softmax", "topk", "entmax", "sparsemax"]


# ── Building blocks ───────────────────────────────────────────────────────────

class FeatureTokenizer(nn.Module):
    """Map each scalar feature to a d-dimensional embedding.

    x_i → W_i · x_i + b_i   where W_i ∈ ℝ^d, b_i ∈ ℝ^d
    """

    def __init__(self, n_features: int, embedding_dim: int) -> None:
        super().__init__()
        # Weight per feature: (n_features, embedding_dim)
        self.weight = nn.Parameter(torch.empty(n_features, embedding_dim))
        self.bias = nn.Parameter(torch.zeros(n_features, embedding_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features)  →  (batch, n_features, embedding_dim)
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class MultiHeadSparseAttention(nn.Module):
    """Multi-head attention with configurable sparsity kernel."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        attn_dropout: float,
        attention_type: AttentionType,
        topk_ratio: float,
        alpha: float,
    ) -> None:
        super().__init__()
        assert embedding_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)

        if attention_type == "topk":
            self.attn_fn = TopKAttention(topk_ratio=topk_ratio)
        elif attention_type == "entmax":
            self.attn_fn = EntmaxAttention(alpha=alpha)
        elif attention_type == "sparsemax":
            self.attn_fn = SparsemaxAttention()
        else:                               # softmax
            self.attn_fn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H, HD = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, L, H, HD).transpose(1, 2)   # (B, H, L, HD)
        k = self.k_proj(x).view(B, L, H, HD).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, HD).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.scale         # (B, H, L, L)

        if self.attn_fn is not None:
            weights = self.attn_fn(scores)
        else:
            weights = F.softmax(scores, dim=-1)

        weights = self.attn_dropout(weights)

        # Record attention weights for sparsity analysis
        self._last_attn_weights = weights.detach()

        out = (weights @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block: LN → Attention → Residual → LN → FFN → Residual."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        attn_dropout: float,
        attention_type: AttentionType,
        topk_ratio: float,
        alpha: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn = MultiHeadSparseAttention(
            embedding_dim, num_heads, attn_dropout, attention_type, topk_ratio, alpha
        )
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ── Main model ────────────────────────────────────────────────────────────────

class FTTransformer(BaseEstimator, ClassifierMixin):
    """FT-Transformer for tabular binary classification.

    Scikit-learn compatible: implements fit / predict / predict_proba.

    Parameters
    ----------
    embedding_dim : int
        Token embedding dimension d.
    num_blocks : int
        Number of Transformer encoder blocks.
    num_heads : int
        Attention heads per block.
    ffn_factor : int
        FFN hidden dim = ffn_factor × embedding_dim.
    dropout : float
        Dropout rate in FFN layers.
    attn_dropout : float
        Dropout rate on attention weights.
    attention_type : str
        One of {"softmax", "topk", "entmax", "sparsemax"}.
    topk_ratio : float
        Fraction of keys kept per query (only for attention_type="topk").
    alpha : float
        Entmax α parameter (only for attention_type="entmax").
    lr : float
        AdamW learning rate.
    weight_decay : float
        AdamW weight decay.
    batch_size : int
        Mini-batch size.
    max_epochs : int
        Maximum training epochs.
    patience : int
        Early stopping patience (on validation loss if val_fraction>0).
    val_fraction : float
        Fraction of training data reserved for early stopping.
    device : str or None
        "cuda" / "cpu" / None (auto-detect).
    random_state : int
        Seed for weight initialisation and data splitting.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        num_blocks: int = 3,
        num_heads: int = 4,
        ffn_factor: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        attention_type: AttentionType = "softmax",
        topk_ratio: float = 0.25,
        alpha: float = 1.5,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
        max_epochs: int = 200,
        patience: int = 20,
        val_fraction: float = 0.10,
        device: str | None = None,
        random_state: int = 42,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.ffn_factor = ffn_factor
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.attention_type = attention_type
        self.topk_ratio = topk_ratio
        self.alpha = alpha
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.device = device
        self.random_state = random_state

    def _get_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self, n_features: int) -> nn.Module:
        d = self.embedding_dim
        ffn_dim = d * self.ffn_factor

        class _Net(nn.Module):
            def __init__(self_, n_features, d, num_blocks, num_heads, ffn_dim,
                         dropout, attn_dropout, attention_type, topk_ratio, alpha):
                super().__init__()
                self_.cls_token = nn.Parameter(torch.zeros(1, 1, d))
                nn.init.normal_(self_.cls_token, std=0.02)
                self_.tokenizer = FeatureTokenizer(n_features, d)
                self_.blocks = nn.ModuleList([
                    TransformerBlock(d, num_heads, ffn_dim, dropout, attn_dropout,
                                     attention_type, topk_ratio, alpha)
                    for _ in range(num_blocks)
                ])
                self_.norm = nn.LayerNorm(d)
                self_.head = nn.Linear(d, 1)

            def forward(self_, x):
                B = x.size(0)
                tokens = self_.tokenizer(x)                          # (B, n_feat, d)
                cls = self_.cls_token.expand(B, -1, -1)             # (B, 1, d)
                tokens = torch.cat([cls, tokens], dim=1)            # (B, n_feat+1, d)
                for block in self_.blocks:
                    tokens = block(tokens)
                cls_out = self_.norm(tokens[:, 0])                  # (B, d)
                return self_.head(cls_out).squeeze(-1)              # (B,)

        return _Net(n_features, d, self.num_blocks, self.num_heads, ffn_dim,
                    self.dropout, self.attn_dropout, self.attention_type,
                    self.topk_ratio, self.alpha)

    def fit(self, X: NDArray, y: NDArray) -> "FTTransformer":
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        torch.manual_seed(self.random_state)

        dev = self._get_device()
        n, p = X.shape
        self.n_features_in_ = p

        # Train / val split for early stopping
        val_n = max(1, int(self.val_fraction * n)) if self.val_fraction > 0 else 0
        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(n)
        val_idx, train_idx = idx[:val_n], idx[val_n:]

        X_t = torch.tensor(X[train_idx], dtype=torch.float32, device=dev)
        y_t = torch.tensor(y[train_idx], dtype=torch.float32, device=dev)
        if val_n > 0:
            X_v = torch.tensor(X[val_idx], dtype=torch.float32, device=dev)
            y_v = torch.tensor(y[val_idx], dtype=torch.float32, device=dev)

        # Remap labels to {0, 1} for BCEWithLogitsLoss
        label_map = {c: float(i) for i, c in enumerate(self.classes_)}
        y_t = torch.tensor([label_map[c.item()] for c in y_t.cpu()],
                            dtype=torch.float32, device=dev)
        if val_n > 0:
            y_v_mapped = torch.tensor([label_map[c.item()] for c in y_v.cpu()],
                                       dtype=torch.float32, device=dev)

        self.model_ = self._build_model(p).to(dev)
        opt = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.BCEWithLogitsLoss()
        n_train = len(train_idx)

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0
        self.train_losses_: list[float] = []

        for epoch in range(self.max_epochs):
            self.model_.train()
            perm = torch.randperm(n_train, device=dev)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_train, self.batch_size):
                batch_idx = perm[start: start + self.batch_size]
                logits = self.model_(X_t[batch_idx])
                loss = loss_fn(logits, y_t[batch_idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                n_batches += 1

            self.train_losses_.append(epoch_loss / max(n_batches, 1))

            # Early stopping on validation loss
            if val_n > 0:
                self.model_.eval()
                with torch.no_grad():
                    val_logits = self.model_(X_v)
                    val_loss = loss_fn(val_logits, y_v_mapped).item()

                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        logger.info("Early stopping at epoch %d (val_loss=%.4f).", epoch, val_loss)
                        break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.n_iter_ = epoch + 1
        return self

    @torch.no_grad()
    def predict_proba(self, X: NDArray) -> NDArray:
        check_is_fitted(self, ["model_"])
        X = check_array(X)
        dev = self._get_device()
        self.model_.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=dev)
        logits = self.model_(X_t).cpu().numpy()
        p1 = 1.0 / (1.0 + np.exp(-logits))    # sigmoid
        return np.column_stack([1 - p1, p1])

    def predict(self, X: NDArray) -> NDArray:
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    def attention_sparsity(self) -> dict:
        """Compute attention sparsity metrics from the last forward pass.

        Returns dict with keys:
            - mean_zero_fraction : average fraction of zero attention weights
            - mean_entropy       : average entropy of attention distributions
            - effective_n_tokens : average effective number of attended tokens
        """
        check_is_fitted(self, ["model_"])
        fracs, entropies, eff_n = [], [], []
        eps = 1e-9

        for block in self.model_.blocks:
            w = block.attn._last_attn_weights   # (B, H, L, L)
            zeros = (w < 1e-4).float().mean().item()
            fracs.append(zeros)

            H = -(w * (w + eps).log()).sum(dim=-1).mean().item()
            entropies.append(H)

            eff = (1.0 / (w**2 + eps).sum(dim=-1)).mean().item()
            eff_n.append(eff)

        return {
            "mean_zero_fraction": float(np.mean(fracs)),
            "mean_entropy": float(np.mean(entropies)),
            "effective_n_tokens": float(np.mean(eff_n)),
        }
