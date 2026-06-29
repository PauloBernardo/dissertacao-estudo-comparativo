"""
Wrapper sklearn-compatível: FT-Transformer + CUR Nyströmformer (colnorm).

Atenção inter-instâncias com aproximação Nyström honesta:
    C = softmax(Q @ K_m^T / √d)  [n × m]  — normalizada sobre m landmark-keys
    R = softmax(Q_m @ K^T  / √d) [m × n]  — normalizada sobre todos n keys
    W = softmax(Q_m @ K_m^T/ √d) [m × m]  — landmark × landmark
    out = C @ (pinv(W) @ (R @ V))          — O(nmd), nunca materializa A n×n

Compressão: m/n ≈ m_ratio → sparsity_ratio = 1 - m/n reportado como
esparsidade de compressão inter-instâncias (comparável ao Nyström-LSSVM).

Nota transductiva
-----------------
    A atenção inter-instâncias requer X_train como contexto na inferência.
    fit() armazena X_train_; predict/predict_proba concatenam
    [X_train_ || X_test] internamente antes de chamar o modelo.
"""

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin

from src.models.landmark_selection import ColumnNormSelector
from src.models.ft_transformer_model import FTTransformerClassifier, fit_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FTTransformerCURColnorm(BaseEstimator, ClassifierMixin):
    """
    FT-Transformer com atenção inter-instâncias aproximada por CUR (colnorm).

    Parâmetros
    ----------
    d_model : int
        Dimensão dos embeddings internos
    n_heads : int
        Número de cabeças de atenção  (deve dividir d_model)
    n_layers : int
        Número de blocos FTBlock (atenção entre features)
    m_ratio : float
        Fração de landmarks CUR  (m = round(m_ratio * n_treino))
    tau_ratio : float
        Limiar para pseudo-inversa truncada  (tau = tau_ratio * σ_max)
    lr : float
        Taxa de aprendizado (Adam)
    epochs : int
        Máximo de épocas de treinamento
    patience : int
        Early stopping: épocas sem melhora na validação
    weight_decay : float
        Regularização L2 do Adam
    val_fraction : float
        Fração de X_train reservada para validação interna
    random_state : int ou None
        Semente para seleção de landmarks e split de validação
    """

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        m_ratio: float = 0.10,
        tau_ratio: float = 0.10,
        lr: float = 5e-4,
        epochs: int = 200,
        patience: int = 20,
        weight_decay: float = 1e-4,
        val_fraction: float = 0.20,
        random_state: int | None = None,
        early_stop_metric: str = "val_acc",
        batch_size: int | None = None,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.m_ratio = m_ratio
        self.tau_ratio = tau_ratio
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.weight_decay = weight_decay
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.early_stop_metric = early_stop_metric
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _to_tensor(self, X: np.ndarray, y: np.ndarray | None = None):
        Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        if y is None:
            return Xt
        yt = torch.tensor(y, dtype=torch.float32, device=DEVICE)
        return Xt, yt

    def _landmark_idx(self, X_train: np.ndarray, m: int) -> torch.Tensor:
        sel = ColumnNormSelector(n_landmarks=m, random_state=self.random_state)
        sel.fit(X_train)
        return torch.tensor(sel.indices_, dtype=torch.long, device=DEVICE)

    # ------------------------------------------------------------------
    # Fit / Predict
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FTTransformerCURColnorm":
        """
        Treina o FT-Transformer com aproximação CUR (colnorm).

        X : (n, d)  float64, já normalizado
        y : (n,)    int  ∈ {0, 1}
        """
        rng = np.random.RandomState(self.random_state)
        n = X.shape[0]
        n_val = max(1, round(self.val_fraction * n))

        # Split interno train / val
        idx_all = rng.permutation(n)
        idx_val, idx_tr = idx_all[:n_val], idx_all[n_val:]
        X_tr, y_tr = X[idx_tr], y[idx_tr]
        X_val, y_val = X[idx_val], y[idx_val]

        # Landmarks sobre o conjunto de treino interno
        m = max(2, round(self.m_ratio * len(idx_tr)))
        self._landmark_idx_ = self._landmark_idx(X_tr, m)

        # Modelo
        self._model = FTTransformerClassifier(
            n_features=X.shape[1],
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            use_inter_instance=True,
            tau_ratio=self.tau_ratio,
            attn_mode="nystrom",          # Nyströmformer: O(nmd), nunca n×n
        ).to(DEVICE)

        X_tr_t, y_tr_t = self._to_tensor(X_tr, y_tr)
        X_val_t, y_val_t = self._to_tensor(X_val, y_val)

        fit_model(
            self._model, X_tr_t, y_tr_t, X_val_t, y_val_t,
            landmark_idx=self._landmark_idx_,
            lr=self.lr, epochs=self.epochs, patience=self.patience,
            weight_decay=self.weight_decay,
            early_stop_metric=self.early_stop_metric,
            batch_size=self.batch_size,
        )

        # Armazena X_train_ completo como contexto para inferência
        # e recalcula landmarks sobre ele (índices agora referem X_train_)
        self.X_train_ = X
        self.y_train_ = y
        m_full = max(2, round(self.m_ratio * n))
        self._landmark_idx_full_ = self._landmark_idx(X, m_full)

        # Esparsidade de compressão inter-instâncias: 1 - m/n
        # Analogia com Nyström-LSSVM: fração de instâncias não usadas como landmark-key
        self.n_landmarks_ = m_full
        self.n_samples_fit_ = n
        self.sparsity_ratio_ = 1.0 - m_full / n
        self.n_support_ = m_full   # landmarks usados como keys

        return self

    @torch.no_grad()
    def _logits(self, X: np.ndarray) -> np.ndarray:
        """Logits para X_test usando X_train_ como contexto."""
        self._model.eval()
        n_ctx = len(self.X_train_)
        X_ctx = np.concatenate([self.X_train_, X], axis=0)
        X_ctx_t = self._to_tensor(X_ctx)
        logits_all = self._model(X_ctx_t, landmark_idx=self._landmark_idx_full_)
        return logits_all[n_ctx:].cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retorna rótulos ∈ {0, 1}."""
        logits = self._logits(X)
        return (1.0 / (1.0 + np.exp(-logits)) >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Probabilidades via sigmoid dos logits.

        Retorna array (n, 2): coluna 0 = P(y=0), coluna 1 = P(y=1).
        """
        logits = self._logits(X)
        p1 = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - p1, p1])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Acurácia (fração de predições corretas)."""
        return float(np.mean(self.predict(X) == y))

    # ------------------------------------------------------------------
    # Registro no runner
    # ------------------------------------------------------------------
    # Em src/experiments/runner.py → _build_model():
    #
    #   elif model_name == "FTTransformerCURColnorm":
    #       from src.models.ft_transformer_cur_wrapper import FTTransformerCURColnorm
    #       return FTTransformerCURColnorm(**model_params), "binary"
    #
    # Em scripts/run_experiments_tier1.py:
    #
    #   RUNNER_TO_TUNING_KEY = {
    #       ..., "FTTransformerCURColnorm": "FTTransformerCURColnorm",
    #   }
    #   DEFAULT_PARAMS = {
    #       ..., "FTTransformerCURColnorm": {
    #               "d_model": 32, "n_heads": 4, "n_layers": 2,
    #               "m_ratio": 0.10, "lr": 5e-4,
    #           },
    #   }
