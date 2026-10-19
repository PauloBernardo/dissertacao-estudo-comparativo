"""FT-LF (Landmark Features): seleção de colunas no estilo CUR aplicada aos TOKENS DE FEATURE.

Não confundir com o FT-CUR (``ft_transformer_cur_wrapper.py``), cuja aproximação Nyström atua
na atenção ENTRE INSTÂNCIAS. Aqui a atenção de cada amostra (CLS + d features) fica restrita a
m colunas-landmark fixadas no fit:

    A = softmax(Q · K_Lᵀ) · V_L,        L = {CLS} ∪ {m features landmark}

As colunas das d − m features restantes são exatamente zero em todas as camadas. Como a cabeça só
lê o CLS e nenhum token landmark atende a um não-landmark, essas features não influenciam a
predição e podem ser REMOVIDAS (tokenizer e entrada) — esparsidade estrutural 1 − m/d, no
sentido do Nyström-LSSVM. Logo o modelo é EXATAMENTE um FT-Transformer softmax sobre as m
colunas escolhidas; é assim que ele é implementado (sem máscara: mesmo resultado, sem custo
das colunas mortas).

Seletores das colunas (sobre a matriz de atenção, como no CUR de Drineas et al.):
    "random"       — uniforme (o que funcionou no Nyström-LSSVM);
    "attn_colnorm" — amostragem sem reposição ∝ ‖A[:, j]‖², com A a atenção de um FT softmax
                     piloto treinado brevemente com todas as features (média em amostras,
                     cabeças e camadas);
    "sparsemax"    — o mesmo, com o piloto em atenção sparsemax (zera colunas de ruído:
                     study_noise_attention_mass mostrou 0,64–0,74 da massa do CLS nas úteis).
"""
from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted

from .ft_transformer import FTTransformer


class FTLandmarkFeatures(BaseEstimator, ClassifierMixin):
    def __init__(self, m_features: int | None = None, m_ratio: float = 0.25,
                 landmark_method: str = "random", pilot_epochs: int = 20,
                 embedding_dim: int = 64, num_blocks: int = 3, num_heads: int = 4,
                 ffn_factor: int = 4, dropout: float = 0.1, attn_dropout: float = 0.0,
                 lr: float = 1e-3, weight_decay: float = 1e-5, batch_size: int = 256,
                 max_epochs: int = 200, patience: int = 20, val_fraction: float = 0.10,
                 device: str | None = None, random_state: int = 42, min_epochs: int = 0):
        self.m_features = m_features
        self.m_ratio = m_ratio
        self.landmark_method = landmark_method
        self.pilot_epochs = pilot_epochs
        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.ffn_factor = ffn_factor
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.device = device
        self.random_state = random_state
        self.min_epochs = min_epochs

    def _ft(self, **over) -> FTTransformer:
        kw = {k: getattr(self, k) for k in (
            "embedding_dim", "num_blocks", "num_heads", "ffn_factor", "dropout", "attn_dropout",
            "lr", "weight_decay", "batch_size", "max_epochs", "patience", "val_fraction",
            "device", "random_state", "min_epochs")}
        return FTTransformer(**{"attention_type": "softmax", **kw, **over})

    def _column_scores(self, X: NDArray, y: NDArray, attention_type: str = "softmax") -> NDArray:
        """‖A[:, j]‖² médio (amostras, cabeças, camadas) de um FT piloto."""
        pilot = self._ft(max_epochs=self.pilot_epochs, patience=self.pilot_epochs,
                         attention_type=attention_type).fit(X, y)
        net = pilot.model_.eval()
        dev = next(net.parameters()).device
        attn = [m for m in net.modules() if hasattr(m, "attn_fn")]
        score = np.zeros(X.shape[1])
        n_seen = 0
        with torch.no_grad():
            for s in range(0, len(X), 256):
                net(torch.tensor(X[s:s + 256], dtype=torch.float32, device=dev))
                for m in attn:
                    w = m._last_attn_weights                  # (B, H, L, L); coluna 0 = CLS
                    score += (w[..., 1:] ** 2).sum(dim=2).sum(dim=(0, 1)).cpu().numpy()
                n_seen += 1
        self.pilot_column_scores_ = score / max(n_seen, 1)
        return self.pilot_column_scores_

    def fit(self, X: NDArray, y: NDArray) -> "FTLandmarkFeatures":
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        d = X.shape[1]
        m = self.m_features if self.m_features is not None else round(self.m_ratio * d)
        m = int(min(max(1, m), d))
        rng = np.random.default_rng(self.random_state)
        if self.landmark_method == "random":
            idx = rng.choice(d, size=m, replace=False)
        elif self.landmark_method in ("attn_colnorm", "sparsemax"):
            p = self._column_scores(X, y, "sparsemax" if self.landmark_method == "sparsemax" else "softmax")
            p = p / p.sum() if p.sum() > 0 else np.full(d, 1.0 / d)
            idx = rng.choice(d, size=m, replace=False, p=p)
        else:
            raise ValueError(f"landmark_method inválido: {self.landmark_method!r}")
        self.landmark_features_ = np.sort(idx)
        self.n_landmarks_ = m
        self.sparsity_ratio_ = 1.0 - m / d
        self.ft_ = self._ft().fit(X[:, self.landmark_features_], y)
        for a in ("best_epoch_", "n_epochs_", "n_steps_", "stopped_early_"):
            if hasattr(self.ft_, a):
                setattr(self, a, getattr(self.ft_, a))
        return self

    def predict_proba(self, X: NDArray) -> NDArray:
        check_is_fitted(self, ["ft_"])
        return self.ft_.predict_proba(np.asarray(X)[:, self.landmark_features_])

    def predict(self, X: NDArray) -> NDArray:
        check_is_fitted(self, ["ft_"])
        return self.ft_.predict(np.asarray(X)[:, self.landmark_features_])
