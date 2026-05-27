"""
Estratégias de Seleção de Landmarks para Aproximação de Matrizes de Interação

Este módulo implementa seis estratégias de seleção de landmarks:
1. Amostragem Aleatória (baseline)
2. Seleção baseada em K-means
3. Seleção baseada em Opposite Maps
4. Amostragem por Leverage Scores
5. Seleção por Norma de Coluna
6. Farthest Point Sampling (FPS)

Autor: Paulo Ricardo Bernardo Silva
Dissertação: Esparsificação Unificada de Matrizes de Interação em LSSVM e Transformers
"""

import numpy as np
from sklearn.cluster import KMeans
from typing import Optional, Tuple
from abc import ABC, abstractmethod


class LandmarkSelector(ABC):
    """Classe base abstrata para seleção de landmarks."""

    def __init__(self, n_landmarks: int, random_state: Optional[int] = None):
        """
        Parâmetros
        ----------
        n_landmarks : int
            Número de landmarks a selecionar (m)
        random_state : int, opcional
            Semente para reprodutibilidade
        """
        self.n_landmarks = n_landmarks
        self.random_state = random_state
        self.indices_ = None

    @abstractmethod
    def fit(self, X: np.ndarray) -> 'LandmarkSelector':
        """
        Seleciona os landmarks a partir dos dados.

        Parâmetros
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Matriz de dados

        Retorna
        -------
        self
        """
        pass

    def get_landmarks(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna os pontos selecionados como landmarks.

        Parâmetros
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Matriz de dados original

        Retorna
        -------
        landmarks : np.ndarray, shape (n_landmarks, n_features)
            Pontos selecionados
        """
        if self.indices_ is None:
            raise ValueError("Selector não foi ajustado. Chame fit() primeiro.")
        return X[self.indices_]


class RandomSelector(LandmarkSelector):
    """
    Seleção de landmarks por amostragem aleatória uniforme.

    Esta é a estratégia baseline mais simples, onde os landmarks
    são selecionados uniformemente ao acaso.
    """

    def fit(self, X: np.ndarray) -> 'RandomSelector':
        """Seleciona m índices aleatoriamente."""
        n_samples = X.shape[0]

        if self.n_landmarks > n_samples:
            raise ValueError(f"n_landmarks ({self.n_landmarks}) > n_samples ({n_samples})")

        rng = np.random.RandomState(self.random_state)
        self.indices_ = rng.choice(n_samples, size=self.n_landmarks, replace=False)
        self.indices_ = np.sort(self.indices_)

        return self


class KMeansSelector(LandmarkSelector):
    """
    Seleção de landmarks baseada em agrupamento K-means.

    Aplica K-means com k = n_landmarks clusters e seleciona
    os pontos mais próximos aos centróides.
    """

    def __init__(self, n_landmarks: int, random_state: Optional[int] = None,
                 n_init: int = 10, max_iter: int = 300):
        """
        Parâmetros
        ----------
        n_landmarks : int
            Número de landmarks (clusters)
        random_state : int, opcional
            Semente para reprodutibilidade
        n_init : int
            Número de inicializações do K-means
        max_iter : int
            Número máximo de iterações
        """
        super().__init__(n_landmarks, random_state)
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans_ = None

    def fit(self, X: np.ndarray) -> 'KMeansSelector':
        """
        Aplica K-means e seleciona pontos mais próximos aos centróides.
        """
        n_samples = X.shape[0]

        if self.n_landmarks > n_samples:
            raise ValueError(f"n_landmarks ({self.n_landmarks}) > n_samples ({n_samples})")

        # Ajustar K-means
        self.kmeans_ = KMeans(
            n_clusters=self.n_landmarks,
            random_state=self.random_state,
            n_init=self.n_init,
            max_iter=self.max_iter
        )
        labels = self.kmeans_.fit_predict(X)
        centroids = self.kmeans_.cluster_centers_

        # Para cada cluster, encontrar o ponto mais próximo ao centróide
        self.indices_ = np.zeros(self.n_landmarks, dtype=int)

        for k in range(self.n_landmarks):
            # Índices dos pontos no cluster k
            cluster_mask = (labels == k)
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) == 0:
                # Cluster vazio: selecionar ponto mais próximo ao centróide globalmente
                distances = np.linalg.norm(X - centroids[k], axis=1)
                self.indices_[k] = np.argmin(distances)
            else:
                # Encontrar ponto mais próximo ao centróide dentro do cluster
                cluster_points = X[cluster_mask]
                distances = np.linalg.norm(cluster_points - centroids[k], axis=1)
                local_idx = np.argmin(distances)
                self.indices_[k] = cluster_indices[local_idx]

        self.indices_ = np.sort(self.indices_)

        return self


class OppositeMapsSelector(LandmarkSelector):
    """
    Seleção de landmarks baseada em Opposite Maps.

    Utiliza a heurística de oposição geométrica para maximizar
    a diversidade do subconjunto selecionado. O operador de
    oposição é definido como:

        x̆_j = a_j + b_j - x_j

    onde a_j e b_j são os limites inferior e superior da dimensão j.

    Referência:
    - Tizhoosh, H.R. (2005). Opposition-Based Learning: A New Scheme
      for Machine Intelligence.
    - Rocha Neto, A.R. & Barreto, G.A. (2013). Opposite Maps: Vector
      Quantization Algorithms for Building Reduced-Set SVM and LSSVM.
    """

    def __init__(self, n_landmarks: int, random_state: Optional[int] = None,
                 use_centroid: bool = False):
        """
        Parâmetros
        ----------
        n_landmarks : int
            Número de landmarks a selecionar
        random_state : int, opcional
            Semente para reprodutibilidade
        use_centroid : bool
            Se True, calcula o oposto do centróide do conjunto selecionado.
            Se False, calcula o oposto do último ponto adicionado.
        """
        super().__init__(n_landmarks, random_state)
        self.use_centroid = use_centroid
        self.bounds_ = None

    def _compute_opposite(self, x: np.ndarray) -> np.ndarray:
        """
        Calcula o ponto oposto de x.

        Parâmetros
        ----------
        x : np.ndarray, shape (n_features,)
            Ponto original

        Retorna
        -------
        x_opposite : np.ndarray, shape (n_features,)
            Ponto oposto
        """
        a, b = self.bounds_
        return a + b - x

    def fit(self, X: np.ndarray) -> 'OppositeMapsSelector':
        """
        Seleciona landmarks usando a heurística de Opposite Maps.

        Algoritmo:
        1. Calcular limites do espaço de dados
        2. Selecionar primeiro landmark aleatoriamente
        3. Para k = 2 até m:
           a. Calcular ponto oposto (do último ponto ou do centróide)
           b. Encontrar ponto mais próximo ao oposto (não selecionado)
           c. Adicionar ao conjunto de landmarks
        """
        n_samples, n_features = X.shape

        if self.n_landmarks > n_samples:
            raise ValueError(f"n_landmarks ({self.n_landmarks}) > n_samples ({n_samples})")

        # Calcular limites do espaço de dados
        a = X.min(axis=0)  # limite inferior por dimensão
        b = X.max(axis=0)  # limite superior por dimensão
        self.bounds_ = (a, b)

        # Inicializar
        rng = np.random.RandomState(self.random_state)
        selected = []
        available = set(range(n_samples))

        # Selecionar primeiro landmark aleatoriamente
        first_idx = rng.choice(list(available))
        selected.append(first_idx)
        available.remove(first_idx)

        # Selecionar os demais landmarks
        for k in range(1, self.n_landmarks):
            if len(available) == 0:
                break

            # Calcular ponto de referência para oposição
            if self.use_centroid:
                # Usar centróide do conjunto selecionado
                reference = X[selected].mean(axis=0)
            else:
                # Usar último ponto adicionado
                reference = X[selected[-1]]

            # Calcular ponto oposto
            opposite = self._compute_opposite(reference)

            # Encontrar ponto mais próximo ao oposto dentre os disponíveis
            available_list = list(available)
            available_points = X[available_list]
            distances = np.linalg.norm(available_points - opposite, axis=1)
            closest_local_idx = np.argmin(distances)
            closest_idx = available_list[closest_local_idx]

            # Adicionar ao conjunto selecionado
            selected.append(closest_idx)
            available.remove(closest_idx)

        self.indices_ = np.array(sorted(selected))

        return self


class QuasiOppositeSelector(LandmarkSelector):
    """
    Seleção de landmarks baseada em Quasi-Oposição.

    Variante do Opposite Maps que usa um ponto intermediário
    entre o centro do espaço e o ponto oposto.

        x̆_j = rand(center_j, opposite_j)

    onde center_j = (a_j + b_j) / 2
    """

    def __init__(self, n_landmarks: int, random_state: Optional[int] = None):
        super().__init__(n_landmarks, random_state)
        self.bounds_ = None

    def fit(self, X: np.ndarray) -> 'QuasiOppositeSelector':
        """Seleciona landmarks usando quasi-oposição."""
        n_samples, n_features = X.shape

        if self.n_landmarks > n_samples:
            raise ValueError(f"n_landmarks ({self.n_landmarks}) > n_samples ({n_samples})")

        # Calcular limites
        a = X.min(axis=0)
        b = X.max(axis=0)
        center = (a + b) / 2
        self.bounds_ = (a, b)

        rng = np.random.RandomState(self.random_state)
        selected = []
        available = set(range(n_samples))

        # Primeiro landmark aleatório
        first_idx = rng.choice(list(available))
        selected.append(first_idx)
        available.remove(first_idx)

        for k in range(1, self.n_landmarks):
            if len(available) == 0:
                break

            # Ponto de referência
            reference = X[selected[-1]]

            # Ponto oposto completo
            opposite = a + b - reference

            # Quasi-oposto: entre centro e oposto
            quasi_opposite = np.zeros(n_features)
            for j in range(n_features):
                low = min(center[j], opposite[j])
                high = max(center[j], opposite[j])
                quasi_opposite[j] = rng.uniform(low, high)

            # Encontrar ponto mais próximo
            available_list = list(available)
            available_points = X[available_list]
            distances = np.linalg.norm(available_points - quasi_opposite, axis=1)
            closest_idx = available_list[np.argmin(distances)]

            selected.append(closest_idx)
            available.remove(closest_idx)

        self.indices_ = np.array(sorted(selected))

        return self


class LeverageScoreSelector(LandmarkSelector):
    """
    Seleção de landmarks por amostragem proporcional a leverage scores.

    Os leverage scores medem a importância estatística de cada linha
    (ou coluna) da matriz na sua decomposição SVD. A probabilidade
    de seleção é proporcional a l_i = sum_j U_k[i,j]^2, onde U_k
    contém os k principais vetores singulares esquerdos.

    Referência:
    - Drineas, P., Mahoney, M.W. & Muthukrishnan, S. (2008).
      Relative-error CUR matrix decompositions.
    """

    def __init__(self, n_landmarks: int, random_state: Optional[int] = None,
                 rank_k: Optional[int] = None, kernel: Optional[object] = None):
        """
        Parâmetros
        ----------
        n_landmarks : int
            Número de landmarks a selecionar
        random_state : int, opcional
            Semente para reprodutibilidade
        rank_k : int, opcional
            Número de componentes singulares para cálculo dos leverage scores.
            Se None, usa min(n_landmarks, 50).
        kernel : callable, opcional
            Função kernel. Se fornecida, calcula K = kernel(X) e faz SVD de K
            em vez de X diretamente (útil para Nyström/LSSVM).
        """
        super().__init__(n_landmarks, random_state)
        self.rank_k = rank_k
        self.kernel = kernel
        self.leverage_scores_ = None

    def fit(self, X: np.ndarray) -> 'LeverageScoreSelector':
        """Seleciona landmarks via amostragem proporcional a leverage scores."""
        n_samples = X.shape[0]

        if self.n_landmarks > n_samples:
            raise ValueError(f"n_landmarks ({self.n_landmarks}) > n_samples ({n_samples})")

        # Determinar a matriz para SVD
        if self.kernel is not None:
            M = self.kernel(X)
        else:
            M = X

        # Determinar rank para truncamento
        k = self.rank_k if self.rank_k is not None else min(self.n_landmarks, 50)
        k = min(k, min(M.shape) - 1) if min(M.shape) > 1 else 1

        # SVD truncada
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        U_k = U[:, :k]

        # Leverage scores: l_i = ||U_k[i, :]||^2
        self.leverage_scores_ = np.sum(U_k ** 2, axis=1)

        # Normalizar para probabilidades
        probs = self.leverage_scores_ / self.leverage_scores_.sum()

        # Tratar probabilidades zero/negativas por erro numérico
        probs = np.maximum(probs, 0)
        probs /= probs.sum()

        # Amostragem sem reposição proporcional a leverage scores
        rng = np.random.RandomState(self.random_state)
        self.indices_ = rng.choice(n_samples, size=self.n_landmarks,
                                    replace=False, p=probs)
        self.indices_ = np.sort(self.indices_)

        return self


class ColumnNormSelector(LandmarkSelector):
    """
    Seleção de landmarks por amostragem proporcional à norma.

    A probabilidade de seleção é proporcional a ||x_i||^2.
    Para CUR, quando as linhas de X são colunas da matriz A,
    isso equivale a amostrar colunas proporcionalmente a ||A[:,j]||^2.

    Referência:
    - Drineas, P., Kannan, R. & Mahoney, M.W. (2006).
      Fast Monte Carlo algorithms for matrices II.
    """

    def __init__(self, n_landmarks: int, random_state: Optional[int] = None,
                 kernel: Optional[object] = None):
        """
        Parâmetros
        ----------
        n_landmarks : int
            Número de landmarks a selecionar
        random_state : int, opcional
            Semente para reprodutibilidade
        kernel : callable, opcional
            Função kernel. Se fornecida, calcula K = kernel(X) e usa normas
            das colunas de K.
        """
        super().__init__(n_landmarks, random_state)
        self.kernel = kernel
        self.norms_ = None

    def fit(self, X: np.ndarray) -> 'ColumnNormSelector':
        """Seleciona landmarks via amostragem proporcional à norma."""
        n_samples = X.shape[0]

        if self.n_landmarks > n_samples:
            raise ValueError(f"n_landmarks ({self.n_landmarks}) > n_samples ({n_samples})")

        # Determinar a matriz para cálculo de normas
        if self.kernel is not None:
            M = self.kernel(X)
        else:
            M = X

        # Norma ao quadrado de cada linha
        self.norms_ = np.sum(M ** 2, axis=1)

        # Normalizar para probabilidades
        total = self.norms_.sum()
        if total < 1e-15:
            # Fallback para uniforme se todas as normas forem zero
            probs = np.ones(n_samples) / n_samples
        else:
            probs = self.norms_ / total

        probs = np.maximum(probs, 0)
        probs /= probs.sum()

        # Amostragem sem reposição
        rng = np.random.RandomState(self.random_state)
        self.indices_ = rng.choice(n_samples, size=self.n_landmarks,
                                    replace=False, p=probs)
        self.indices_ = np.sort(self.indices_)

        return self


class FarthestPointSelector(LandmarkSelector):
    """
    Seleção de landmarks por Farthest Point Sampling (FPS).

    Algoritmo guloso que seleciona iterativamente o ponto mais
    distante do conjunto já selecionado (distância ao vizinho
    mais próximo no conjunto selecionado).

    Referência:
    - Eldar, Y., Lindenbaum, M., Porat, M. & Zeevi, Y.Y. (1997).
      The farthest point strategy for progressive image sampling.
    """

    def __init__(self, n_landmarks: int, random_state: Optional[int] = None):
        super().__init__(n_landmarks, random_state)

    def fit(self, X: np.ndarray) -> 'FarthestPointSelector':
        """Seleciona landmarks por Farthest Point Sampling."""
        n_samples = X.shape[0]

        if self.n_landmarks > n_samples:
            raise ValueError(f"n_landmarks ({self.n_landmarks}) > n_samples ({n_samples})")

        rng = np.random.RandomState(self.random_state)

        # Primeiro ponto aleatório
        first_idx = rng.randint(n_samples)
        selected = [first_idx]

        # Distância mínima de cada ponto ao conjunto selecionado
        min_dists = np.full(n_samples, np.inf)
        # Atualizar com distâncias ao primeiro ponto
        diffs = X - X[first_idx]
        min_dists = np.sum(diffs ** 2, axis=1)
        min_dists[first_idx] = -1.0  # Marcar como selecionado

        for _ in range(1, self.n_landmarks):
            # Selecionar o ponto com maior distância mínima
            next_idx = np.argmax(min_dists)
            selected.append(next_idx)
            min_dists[next_idx] = -1.0  # Marcar como selecionado

            # Atualizar distâncias mínimas
            diffs = X - X[next_idx]
            new_dists = np.sum(diffs ** 2, axis=1)
            min_dists = np.where(min_dists >= 0,
                                  np.minimum(min_dists, new_dists),
                                  min_dists)

        self.indices_ = np.array(sorted(selected))

        return self


def get_selector(method: str, n_landmarks: int, random_state: Optional[int] = None,
                 **kwargs) -> LandmarkSelector:
    """
    Factory function para criar seletores de landmarks.

    Parâmetros
    ----------
    method : str
        Método de seleção: 'random', 'kmeans', 'opposite', 'quasi_opposite'
    n_landmarks : int
        Número de landmarks
    random_state : int, opcional
        Semente para reprodutibilidade
    **kwargs
        Argumentos adicionais para o seletor específico

    Retorna
    -------
    selector : LandmarkSelector
        Instância do seletor apropriado
    """
    methods = {
        'random': RandomSelector,
        'kmeans': KMeansSelector,
        'opposite': OppositeMapsSelector,
        'quasi_opposite': QuasiOppositeSelector,
        'leverage': LeverageScoreSelector,
        'colnorm': ColumnNormSelector,
        'fps': FarthestPointSelector,
    }

    if method not in methods:
        raise ValueError(f"Método desconhecido: {method}. "
                        f"Opções: {list(methods.keys())}")

    return methods[method](n_landmarks, random_state, **kwargs)


if __name__ == "__main__":
    # Exemplo de uso
    from sklearn.datasets import make_classification

    # Gerar dados sintéticos
    X, y = make_classification(n_samples=500, n_features=10,
                               n_informative=5, random_state=42)

    n_landmarks = 50  # 10% dos dados

    print("Teste das estratégias de seleção de landmarks")
    print("=" * 50)
    print(f"Dados: {X.shape[0]} amostras, {X.shape[1]} features")
    print(f"Landmarks: {n_landmarks}")
    print()

    for method in ['random', 'kmeans', 'opposite', 'quasi_opposite',
                    'leverage', 'colnorm', 'fps']:
        selector = get_selector(method, n_landmarks, random_state=42)
        selector.fit(X)

        landmarks = selector.get_landmarks(X)
        print(f"{method:15s}: {len(selector.indices_)} landmarks selecionados")
        print(f"                Índices: {selector.indices_[:10]}...")
        print()
