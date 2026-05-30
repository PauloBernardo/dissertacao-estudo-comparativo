"""
Wrapper sklearn-compatível: SAINT (Self-Attention and Intersample Attention Transformer).

SAINT alterna blocos de atenção inter-features e inter-instâncias em cada camada,
usando atenção softmax completa (n×n) na dimensão inter-instâncias — sem aproximação.

É o baseline denso para comparação com o FT-CUR Nyströmformer:
  SAINT      : inter-instance softmax completo   (O(n²d) — denso)
  FT-CUR     : inter-instance Nyströmformer       (O(nmd) — comprimido, m << n)

Nota transductiva
-----------------
    A atenção inter-instâncias requer X_train como contexto na inferência.
    fit() armazena X_train_; predict/predict_proba concatenam
    [X_train_ || X_test] internamente.

Esparsidade
-----------
    Atenção inter-instâncias: softmax → sem zeros exatos → sparsity_ratio = 0.
    Reportado como "inter-instance attention" com 0% de compressão.
"""

import numpy as np
import torch

from src.models.ft_transformer_model import SAINTClassifier, fit_model


# SAINT mantém a atenção inter-instâncias n×n completa.
# Em GPUs pequenas (ex.: MX350 com 2GB) o forward/backward pode estourar
# para N≥3000. Detectamos automaticamente: usa CUDA se houver ≥4GB livres,
# senão CPU. T4 (16GB) e A100 (40GB) cabem confortavelmente.
def _pick_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes >= 4 * 1024**3:  # ≥4GB livres
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = _pick_device()


class SAINTColnorm:
    """
    SAINT para classificação binária — baseline de atenção inter-instâncias completa.

    Parâmetros
    ----------
    d_model : int
    n_heads : int
    n_layers : int
    lr : float
    epochs : int
    patience : int
    weight_decay : float
    val_fraction : float
    random_state : int | None
    """

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        lr: float = 5e-4,
        epochs: int = 200,
        patience: int = 20,
        weight_decay: float = 1e-4,
        val_fraction: float = 0.20,
        random_state: int | None = None,
        early_stop_metric: str = "val_acc",
    ):
        self.d_model      = d_model
        self.n_heads      = n_heads
        self.n_layers     = n_layers
        self.lr           = lr
        self.epochs       = epochs
        self.patience     = patience
        self.weight_decay = weight_decay
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.early_stop_metric = early_stop_metric

    def _to_tensor(self, X, y=None):
        Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        if y is None:
            return Xt
        return Xt, torch.tensor(y, dtype=torch.float32, device=DEVICE)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SAINTColnorm":
        rng = np.random.RandomState(self.random_state)
        n = X.shape[0]
        n_val = max(1, round(self.val_fraction * n))

        idx = rng.permutation(n)
        X_val, X_tr = X[idx[:n_val]], X[idx[n_val:]]
        y_val, y_tr = y[idx[:n_val]], y[idx[n_val:]]

        self._model = SAINTClassifier(
            n_features=X.shape[1],
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
        ).to(DEVICE)

        X_tr_t, y_tr_t = self._to_tensor(X_tr, y_tr)
        X_val_t, y_val_t = self._to_tensor(X_val, y_val)

        fit_model(
            self._model, X_tr_t, y_tr_t, X_val_t, y_val_t,
            landmark_idx=None,          # SAINT: atenção completa, sem landmarks
            lr=self.lr, epochs=self.epochs, patience=self.patience,
            weight_decay=self.weight_decay,
            early_stop_metric=self.early_stop_metric,
        )

        self.X_train_ = X
        self.y_train_ = y

        # Atenção inter-instâncias completa → sem compressão
        self.n_landmarks_   = n          # todos são "landmarks"
        self.n_samples_fit_ = n
        self.sparsity_ratio_ = 0.0       # softmax denso, sem zeros exatos
        self.n_support_      = n

        return self

    @torch.no_grad()
    def _logits(self, X: np.ndarray) -> np.ndarray:
        self._model.eval()
        n_ctx = len(self.X_train_)
        X_ctx = np.concatenate([self.X_train_, X], axis=0)
        X_ctx_t = self._to_tensor(X_ctx)
        logits_all = self._model(X_ctx_t, landmark_idx=None)
        return logits_all[n_ctx:].cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (1.0 / (1.0 + np.exp(-self._logits(X))) >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = 1.0 / (1.0 + np.exp(-self._logits(X)))
        return np.column_stack([1.0 - p1, p1])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))
