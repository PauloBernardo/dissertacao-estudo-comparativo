"""
Wrapper sklearn-compatível: LSSVM + Nyström com colnorm (m/n=20%).

Dependências (copiar junto para o outro projeto):
    nystrom.py
    landmark_selection.py

Interface
---------
    y ∈ {-1, +1}   (signed)
    predict_proba retorna sigmoid da função de decisão

Uso mínimo
----------
    from nystrom_lssvm_wrapper import NystromLSSVMColnorm
    model = NystromLSSVMColnorm(sigma=1.0, gamma=1.0)
    model.fit(X_train, y_train)          # y ∈ {-1, +1}
    preds  = model.predict(X_test)
    probas = model.predict_proba(X_test) # shape (n, 2)
"""

import numpy as np

from src.models.nystrom import NystromLSSVM, RBFKernel


class NystromLSSVMColnorm:
    """
    LSSVM + Nyström com seleção de landmarks por norma de coluna (colnorm).

    Parâmetros
    ----------
    sigma : float
        Largura do kernel RBF  (gamma_rbf = 1 / (2 * sigma²))
    gamma : float
        Regularização do LSSVM  (λ = 1/gamma)
    m_ratio : float
        Fração de landmarks  (m = round(m_ratio * n_treino))
    random_state : int ou None
        Semente para a amostragem colnorm
    """

    def __init__(self, sigma: float = 1.0, gamma: float = 1.0,
                 m_ratio: float = 0.20, random_state: int | None = None):
        self.sigma = sigma
        self.gamma = gamma
        self.m_ratio = m_ratio
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Fit / Predict
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NystromLSSVMColnorm":
        """
        Treina o LSSVM com aproximação Nyström (colnorm).

        X : (n, d)  float64, já normalizado
        y : (n,)    int  ∈ {-1, +1}
        """
        n = X.shape[0]
        m = max(2, round(self.m_ratio * n))
        kernel = RBFKernel(sigma=self.sigma)

        self._model = NystromLSSVM(
            n_landmarks=m,
            gamma=self.gamma,
            kernel=kernel,
            selection_method="colnorm",
            random_state=self.random_state,
        )
        self._model.fit(X, y.astype(float))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retorna rótulos ∈ {-1, +1}."""
        return self._model.predict(X).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Probabilidades via sigmoid da função de decisão.

        Retorna array (n, 2): coluna 0 = P(y=-1), coluna 1 = P(y=+1).
        """
        scores = self._model.decision_function(X)
        p1 = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack([1.0 - p1, p1])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Acurácia (fração de predições corretas)."""
        return float(np.mean(self.predict(X) == y))

    # ------------------------------------------------------------------
    # Esparsidade (opcional — runner coleta automaticamente)
    # ------------------------------------------------------------------

    @property
    def support_vectors_(self) -> np.ndarray:
        """Índices dos landmarks selecionados no conjunto de treino."""
        return self._model.nystrom_.selector_.indices_

    @property
    def n_support_(self) -> int:
        return int(len(self.support_vectors_))

    @property
    def n_samples_fit_(self) -> int:
        return int(self._model.X_train_.shape[0])

    @property
    def sparsity_ratio_(self) -> float:
        """Fração de amostras NÃO usadas como landmarks (1 - m/n)."""
        return 1.0 - self.n_support_ / self.n_samples_fit_

    # ------------------------------------------------------------------
    # Registro no runner
    # ------------------------------------------------------------------
    # Em src/experiments/runner.py → _build_model():
    #
    #   elif model_name == "NystromLSSVMColnorm":
    #       from src.models.nystrom_lssvm_wrapper import NystromLSSVMColnorm
    #       return NystromLSSVMColnorm(**model_params), "signed"
    #
    # Em scripts/run_experiments_tier1.py:
    #
    #   RUNNER_TO_TUNING_KEY = {
    #       ..., "NystromLSSVMColnorm": "NystromLSSVMColnorm",
    #   }
    #   DEFAULT_PARAMS = {
    #       ..., "NystromLSSVMColnorm": {"sigma": 1.0, "gamma": 1.0, "m_ratio": 0.20},
    #   }
    #   _LSSVM_MODELS = {"StandardLSSVM", ..., "NystromLSSVMColnorm"}
