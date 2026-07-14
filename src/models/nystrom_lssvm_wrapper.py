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

Paper-fonte (BASE TEORICA, fora do repo — ver docs/model_references.md):
    LSSVM/CLASSICOS/NIPS-2000-using-the-nystrom-method-to-speed-up-kernel-machines-Paper.pdf
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from src.models.nystrom import NystromLSSVM, RBFKernel


class NystromLSSVMColnorm(BaseEstimator, ClassifierMixin):
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
        self.is_fitted_ = True
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
        return self._model.nystrom_.indices_

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


class NystromLSSVMRandom(NystromLSSVMColnorm):
    """
    LSSVM + Nyström com seleção de landmarks por amostragem aleatória
    uniforme (baseline).

    Idêntico ao :class:`NystromLSSVMColnorm` em todos os aspectos exceto
    o método de seleção de landmarks, que aqui é ``random``. Serve como
    baseline para isolar o efeito do seletor colnorm sobre a acurácia,
    mantendo kernel, regularização e orçamento de landmarks (m_ratio)
    idênticos.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NystromLSSVMRandom":
        n = X.shape[0]
        m = max(2, round(self.m_ratio * n))
        kernel = RBFKernel(sigma=self.sigma)

        self._model = NystromLSSVM(
            n_landmarks=m,
            gamma=self.gamma,
            kernel=kernel,
            selection_method="random",
            random_state=self.random_state,
        )
        self._model.fit(X, y.astype(float))
        self.is_fitted_ = True
        return self


class NystromLSSVMKmeans(NystromLSSVMColnorm):
    """
    LSSVM + Nyström com seleção de landmarks por K-means (protótipos = pontos
    mais próximos aos centróides). Baseline de seleção "inteligente" usado como
    referência na literatura (ex.: paper ICML de Nyström). Idêntico ao
    :class:`NystromLSSVMColnorm` exceto pelo seletor.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NystromLSSVMKmeans":
        n = X.shape[0]
        m = max(2, round(self.m_ratio * n))
        kernel = RBFKernel(sigma=self.sigma)

        self._model = NystromLSSVM(
            n_landmarks=m,
            gamma=self.gamma,
            kernel=kernel,
            selection_method="kmeans",
            random_state=self.random_state,
        )
        self._model.fit(X, y.astype(float))
        self.is_fitted_ = True
        return self


def _kernel_kmeans_medoids(K: np.ndarray, k: int, rng: np.random.Generator,
                           max_iter: int = 20) -> np.ndarray:
    """Kernel k-means EM FEATURE SPACE (VQ fiel do OM original, variante K2M).

    Agrupa os pontos cuja matriz kernel é ``K`` em ``k`` clusters e retorna os
    índices dos medoides (ponto mais próximo do centróide em feature space) dos
    clusters não-vazios. Vetorizado via BLAS:
        d²(i,j) = K(i,i) − 2·(K M)[i,j]/|Rⱼ| + diag(Mᵀ K M)[j]/|Rⱼ|².
    """
    m = K.shape[0]
    k = max(1, min(k, m))
    diag = np.diag(K)
    labels = rng.integers(0, k, size=m)

    def _d2(labels: np.ndarray) -> np.ndarray:
        M = np.zeros((m, k))
        M[np.arange(m), labels] = 1.0
        sz = M.sum(0)
        KM = K @ M
        h = np.einsum("ij,ij->j", M, KM)
        with np.errstate(divide="ignore", invalid="ignore"):
            d2 = diag[:, None] - 2.0 * KM / sz[None, :] + (h / sz ** 2)[None, :]
        d2[:, sz == 0] = np.inf
        return d2

    for _ in range(max_iter):
        new_labels = np.argmin(_d2(labels), axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

    d2 = _d2(labels)
    medoids: list[int] = []
    for j in range(k):
        members = np.where(labels == j)[0]
        if members.size == 0:
            continue
        medoids.append(int(members[np.argmin(d2[members, j])]))
    return np.array(sorted(set(medoids)), dtype=int)


def select_opposite_landmarks(X: np.ndarray, y: np.ndarray, sigma: float,
                              m_ratio: float, random_state=None) -> np.ndarray:
    """Seleção de landmarks pela ideia do Opposite Maps original (K2M) a
    orçamento fixo $m = \\text{m\\_ratio}\\cdot n$ (ver :class:`NystromLSSVMOpposite`).

    Supervisionada e dependente de $\\sigma$: super-gera $2m$ medoides por
    kernel k-means em feature space por classe, depois poda pelo mapa de
    oposição (votos da classe contrária) até os $m$ de maior fronteira.
    Reutilizada pelo Nyström-SVM e pelo FT-CUR (com $\\sigma$ por heurística
    da mediana, já que a atenção do FT-CUR não expõe um $\\sigma$).
    """
    n = X.shape[0]
    m = max(2, round(m_ratio * n))
    kernel = RBFKernel(sigma=sigma)
    rng = np.random.default_rng(random_state)

    pos_idx = np.where(y > 0)[0]
    neg_idx = np.where(y <= 0)[0]

    cand = []
    for cls_idx in (pos_idx, neg_idx):
        if len(cls_idx) == 0:
            continue
        k_c = max(1, min(len(cls_idx), round(2 * m_ratio * len(cls_idx))))
        K_c = kernel(X[cls_idx])
        med = _kernel_kmeans_medoids(K_c, k_c, rng)
        cand.extend(cls_idx[med].tolist())
    cand = np.array(sorted(set(cand)), dtype=int)

    if len(cand) <= m:
        return cand

    cand_pos_mask = np.isin(cand, pos_idx)
    votes = np.zeros(len(cand))
    boundary = np.full(len(cand), -np.inf)
    pos_cand_local = np.where(cand_pos_mask)[0]
    neg_cand_local = np.where(~cand_pos_mask)[0]

    if len(pos_cand_local) > 0 and len(neg_idx) > 0:
        Kpc = kernel(X[neg_idx], X[cand[pos_cand_local]])
        cnt = np.bincount(np.argmax(Kpc, axis=1), minlength=len(pos_cand_local))
        votes[pos_cand_local] += cnt
        boundary[pos_cand_local] = Kpc.max(axis=0)
    if len(neg_cand_local) > 0 and len(pos_idx) > 0:
        Knc = kernel(X[pos_idx], X[cand[neg_cand_local]])
        cnt = np.bincount(np.argmax(Knc, axis=1), minlength=len(neg_cand_local))
        votes[neg_cand_local] += cnt
        boundary[neg_cand_local] = Knc.max(axis=0)

    order = np.lexsort((boundary, votes))[::-1]
    return cand[np.sort(order[:m])]


def median_heuristic_sigma(X: np.ndarray, max_samples: int = 1000,
                           random_state=None) -> float:
    """$\\sigma$ pela heurística da mediana: mediana das distâncias par a par
    (subamostradas para custo controlado). Usada pelo seletor opposite do
    FT-CUR, que não expõe um $\\sigma$ próprio."""
    from scipy.spatial.distance import pdist
    rng = np.random.default_rng(random_state)
    if len(X) > max_samples:
        X = X[rng.choice(len(X), max_samples, replace=False)]
    d = pdist(X)
    med = float(np.median(d)) if d.size else 1.0
    return med if med > 1e-9 else 1.0


class NystromLSSVMOpposite(NystromLSSVMColnorm):
    """
    LSSVM + Nyström com seleção de landmarks pela ideia do Opposite Maps
    original (Neto & Barreto 2013, variante K2M), adaptada a um orçamento fixo
    de landmarks ($m = \\text{m\\_ratio} \\cdot n$).

    Algoritmo (supervisionado, depende de $\\sigma$):
      1. Super-geração: kernel k-means EM FEATURE SPACE por classe, com
         $k_c = \\text{round}(2\\,\\text{m\\_ratio}\\,n_c)$ clusters → medoides
         candidatos (~2× o orçamento final).
      2. Poda por opposite-map: cada ponto da classe oposta vota no seu
         protótipo mais próximo (maior kernel = mais perto em feature space).
         Os candidatos são ranqueados por nº de votos (relevância de fronteira,
         essência do opposite maps; desempate por proximidade máxima à classe
         oposta) e mantêm-se os top-$m$.

    NÃO confundir com OppositeMapsLSSVM/OppositeMapsOriginalLSSVM (modelos
    reduced-set standalone); aqui é apenas o seletor de landmarks do Nyström.
    """

    def _select_landmarks(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return select_opposite_landmarks(
            X, y, sigma=self.sigma, m_ratio=self.m_ratio,
            random_state=self.random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NystromLSSVMOpposite":
        y = y.astype(float)
        landmark_indices = self._select_landmarks(X, y)
        kernel = RBFKernel(sigma=self.sigma)

        self._model = NystromLSSVM(
            n_landmarks=len(landmark_indices),
            gamma=self.gamma,
            kernel=kernel,
            landmark_indices=landmark_indices,
            random_state=self.random_state,
        )
        self._model.fit(X, y)
        self.is_fitted_ = True
        return self
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
