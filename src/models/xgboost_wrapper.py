"""
Wrapper sklearn-compatível para XGBoost — baseline tree-based geral.

Usado como referência de "estado da arte tabular" para contextualizar
LSSVMs e Transformers. XGBoost (Chen & Guestrin, 2016) é amplamente
documentado e considerado padrão-ouro em dados tabulares estruturados.

Esparsidade
-----------
    XGBoost produz ensembles de árvores que naturalmente selecionam features.
    Não é o mesmo tipo de "esparsidade" dos LSSVMs (de amostras) nem dos
    Transformers (de atenção); reportamos como N/A no contexto deste estudo.
"""

import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin


class XGBoostBaseline(BaseEstimator, ClassifierMixin):
    """
    XGBoost para classificação binária.

    Parâmetros
    ----------
    n_estimators : int
    max_depth : int
    learning_rate : float
    subsample : float
    colsample_bytree : float
    reg_lambda : float
    reg_alpha : float
    random_state : int | None
    n_jobs : int
        Threads do XGBoost. 1 na v1 (o GridSearchCV já paraleliza os folds);
        -1 no Tier 3, cujos trials são sequenciais. ``hist`` é determinístico
        com várias threads: muda o tempo, não o modelo.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        random_state: int | None = None,
        n_jobs: int = 1,
    ):
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.learning_rate    = learning_rate
        self.subsample        = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda       = reg_lambda
        self.reg_alpha        = reg_alpha
        self.random_state     = random_state
        self.n_jobs           = n_jobs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostBaseline":
        self._model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            random_state=self.random_state,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=self.n_jobs,
            verbosity=0,
        )
        self._model.fit(X, y.astype(int))
        self.n_samples_fit_ = len(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))
