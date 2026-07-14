"""
Aproximação de Nyström para Matrizes Kernel

Este módulo implementa a aproximação de Nyström para reduzir
a complexidade computacional de métodos baseados em kernel.

K̃ = C W^{-1} C^T

onde:
- C ∈ R^{n×m}: colunas da matriz kernel correspondentes aos landmarks
- W ∈ R^{m×m}: submatriz entre landmarks (W = K[landmarks, landmarks])

Autor: Paulo Ricardo Bernardo Silva
Dissertação: Esparsificação Unificada de Matrizes de Interação em LSSVM e Transformers
"""

import numpy as np
from scipy.spatial.distance import cdist
from typing import Optional, Callable, Tuple
from src.models.landmark_selection import LandmarkSelector, get_selector


class RBFKernel:
    """Kernel RBF (Gaussiano)."""

    def __init__(self, sigma: float = 1.0):
        """
        Parâmetros
        ----------
        sigma : float
            Largura do kernel (bandwidth)
        """
        self.sigma = sigma
        self.gamma = 1.0 / (2 * sigma ** 2)

    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calcula a matriz kernel K(X, Y).

        K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
        """
        if Y is None:
            Y = X
        sq_dists = cdist(X, Y, metric='sqeuclidean')
        return np.exp(-self.gamma * sq_dists)


class LinearKernel:
    """Kernel Linear."""

    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """K(x, y) = x^T y"""
        if Y is None:
            Y = X
        return X @ Y.T


class PolynomialKernel:
    """Kernel Polinomial."""

    def __init__(self, degree: int = 3, coef0: float = 1.0):
        self.degree = degree
        self.coef0 = coef0

    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """K(x, y) = (x^T y + coef0)^degree"""
        if Y is None:
            Y = X
        return (X @ Y.T + self.coef0) ** self.degree


class NystromApproximation:
    """
    Aproximação de Nyström para matrizes kernel.

    A aproximação de Nyström reduz a complexidade de O(n²) para O(nm),
    onde m é o número de landmarks.

    Referências:
    - Williams & Seeger (2001). Using the Nyström Method to Speed Up Kernel Machines.
    - Drineas & Mahoney (2005). On the Nyström Method for Approximating a Gram Matrix.
    """

    def __init__(self, n_landmarks: int, kernel: Callable = None,
                 selection_method: str = 'random',
                 random_state: Optional[int] = None,
                 regularization: float = 1e-10,
                 landmark_indices: Optional[np.ndarray] = None,
                 **selector_kwargs):
        """
        Parâmetros
        ----------
        n_landmarks : int
            Número de landmarks (m)
        kernel : callable, opcional
            Função kernel. Se None, usa RBF com sigma=1.0
        selection_method : str
            Método de seleção: 'random', 'kmeans', 'opposite', 'quasi_opposite'
        random_state : int, opcional
            Semente para reprodutibilidade
        regularization : float
            Termo de regularização para inversão de W (estabilidade numérica)
        **selector_kwargs
            Argumentos adicionais para o seletor de landmarks
        """
        self.n_landmarks = n_landmarks
        self.kernel = kernel if kernel is not None else RBFKernel(sigma=1.0)
        self.selection_method = selection_method
        self.random_state = random_state
        self.regularization = regularization
        self.landmark_indices = landmark_indices
        self.selector_kwargs = selector_kwargs

        # Atributos preenchidos após fit
        self.selector_ = None
        self.indices_ = None
        self.landmarks_ = None
        self.W_ = None
        self.W_inv_ = None
        self.C_ = None

    def fit(self, X: np.ndarray) -> 'NystromApproximation':
        """
        Ajusta a aproximação de Nyström.

        Parâmetros
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Dados de treinamento

        Retorna
        -------
        self
        """
        n_samples = X.shape[0]

        if self.landmark_indices is not None:
            # Landmarks pré-computados (ex.: seleção supervisionada por
            # kernel k-means em feature space no wrapper Opposite). Ignora o
            # seletor unsupervised e usa os índices fornecidos diretamente.
            self.selector_ = None
            self.indices_ = np.asarray(self.landmark_indices, dtype=int)
            self.n_landmarks = len(self.indices_)
            self.landmarks_ = X[self.indices_]
        else:
            # Selecionar landmarks
            kwargs = dict(self.selector_kwargs)
            # 'leverage' needs the kernel; 'colnorm' uses feature norms (||x_i||²)
            # to avoid computing the full N×N kernel matrix (O(n²) memory).
            if self.selection_method == 'leverage':
                kwargs.setdefault('kernel', self.kernel)
            self.selector_ = get_selector(
                self.selection_method,
                self.n_landmarks,
                self.random_state,
                **kwargs
            )
            self.selector_.fit(X)
            self.indices_ = self.selector_.indices_
            # Obter landmarks
            self.landmarks_ = self.selector_.get_landmarks(X)

        # Calcular W = K(landmarks, landmarks)
        self.W_ = self.kernel(self.landmarks_, self.landmarks_)

        # Inverter W com regularização para estabilidade numérica
        W_reg = self.W_ + self.regularization * np.eye(self.n_landmarks)
        self.W_inv_ = np.linalg.inv(W_reg)

        # Calcular C = K(X, landmarks)
        self.C_ = self.kernel(X, self.landmarks_)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula a matriz kernel aproximada K̃(X, X_train).

        Para novos dados X, calcula a aproximação da matriz kernel
        entre X e os dados de treinamento originais.

        Parâmetros
        ----------
        X : np.ndarray, shape (n_samples_new, n_features)
            Novos dados

        Retorna
        -------
        K_approx : np.ndarray, shape (n_samples_new, n_samples_train)
            Matriz kernel aproximada
        """
        if self.landmarks_ is None:
            raise ValueError("Approximation não foi ajustada. Chame fit() primeiro.")

        # K_new = K(X_new, landmarks)
        K_new = self.kernel(X, self.landmarks_)

        # K̃ = K_new @ W^{-1} @ C^T
        return K_new @ self.W_inv_ @ self.C_.T

    def get_approximation(self) -> np.ndarray:
        """
        Retorna a matriz kernel aproximada K̃ para os dados de treinamento.

        K̃ = C @ W^{-1} @ C^T

        Retorna
        -------
        K_approx : np.ndarray, shape (n_samples, n_samples)
            Matriz kernel aproximada
        """
        if self.C_ is None:
            raise ValueError("Approximation não foi ajustada. Chame fit() primeiro.")

        return self.C_ @ self.W_inv_ @ self.C_.T

    def approximation_error(self, K_true: np.ndarray) -> Tuple[float, float]:
        """
        Calcula o erro de aproximação.

        Parâmetros
        ----------
        K_true : np.ndarray
            Matriz kernel verdadeira

        Retorna
        -------
        frobenius_error : float
            Erro relativo na norma de Frobenius: ||K - K̃||_F / ||K||_F
        spectral_error : float
            Erro relativo na norma espectral: ||K - K̃||_2 / ||K||_2
        """
        K_approx = self.get_approximation()
        diff = K_true - K_approx

        # Erro de Frobenius (relativo)
        frobenius_error = np.linalg.norm(diff, 'fro') / np.linalg.norm(K_true, 'fro')

        # Erro espectral (relativo)
        spectral_error = np.linalg.norm(diff, 2) / np.linalg.norm(K_true, 2)

        return frobenius_error, spectral_error


class NystromLSSVM:
    """
    LSSVM com aproximação de Nyström.

    Resolve o sistema KKT do LSSVM usando a matriz kernel aproximada,
    reduzindo a complexidade de O(n³) para O(nm² + m³).
    """

    def __init__(self, n_landmarks: int, gamma: float = 1.0,
                 kernel: Callable = None, selection_method: str = 'random',
                 random_state: Optional[int] = None,
                 landmark_indices: Optional[np.ndarray] = None,
                 **selector_kwargs):
        """
        Parâmetros
        ----------
        n_landmarks : int
            Número de landmarks
        gamma : float
            Parâmetro de regularização do LSSVM
        kernel : callable, opcional
            Função kernel
        selection_method : str
            Método de seleção de landmarks
        random_state : int, opcional
            Semente para reprodutibilidade
        """
        self.n_landmarks = n_landmarks
        self.gamma = gamma
        self.kernel = kernel if kernel is not None else RBFKernel(sigma=1.0)
        self.selection_method = selection_method
        self.random_state = random_state
        self.landmark_indices = landmark_indices
        self.selector_kwargs = selector_kwargs

        # Atributos após fit
        self.nystrom_ = None
        self.alpha_ = None
        self.b_ = None
        self.X_train_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NystromLSSVM':
        """
        Treina o LSSVM com aproximação de Nyström.

        Usa a identidade de Woodbury para resolver o sistema KKT em O(nm²)
        sem nunca formar a matriz N×N, mantendo uso de memória em O(nm).

        Sistema KKT:  [0   1ᵀ ] [b    ]   [0]
                      [1   Ω  ] [alpha] = [y]
        onde Ω = C W⁻¹ Cᵀ + (1/γ)I,  C ∈ ℝⁿˣᵐ,  W ∈ ℝᵐˣᵐ.

        Woodbury: Ω⁻¹ rhs = γ·rhs − γ²·C·(W + γ CᵀC)⁻¹·Cᵀ·rhs
        Bias:     b = (1ᵀ Ω⁻¹ y) / (1ᵀ Ω⁻¹ 1)
        Alpha:    α = Ω⁻¹ y − b · Ω⁻¹ 1
        """
        n_samples = X.shape[0]
        self.X_train_ = X

        # Ajustar aproximação de Nyström — produz C (n×m) e W_inv (m×m)
        self.nystrom_ = NystromApproximation(
            n_landmarks=self.n_landmarks,
            kernel=self.kernel,
            selection_method=self.selection_method,
            random_state=self.random_state,
            landmark_indices=self.landmark_indices,
            **self.selector_kwargs
        )
        self.nystrom_.fit(X)

        C     = self.nystrom_.C_      # (n, m)
        W_inv = self.nystrom_.W_inv_  # (m, m)
        gamma = self.gamma

        # H = W + γ CᵀC   (m×m — o único sistema a resolver)
        W_reg = np.linalg.inv(W_inv)          # m×m: custo O(m³), m ≪ n
        H     = W_reg + gamma * (C.T @ C)

        def _woodbury(rhs: np.ndarray) -> np.ndarray:
            """Ω⁻¹ rhs sem formar Ω."""
            return gamma * rhs - gamma ** 2 * C @ np.linalg.solve(H, C.T @ rhs)

        v = _woodbury(y)                 # Ω⁻¹ y
        w = _woodbury(np.ones(n_samples))  # Ω⁻¹ 1

        # Bias e alpha via KKT
        self.b_     = float(np.sum(v) / np.sum(w))
        self.alpha_ = v - self.b_ * w

        # Cache m-vetor para predição O(n_test × m): W⁻¹ Cᵀ α
        self._w_alpha_ = W_inv @ (C.T @ self.alpha_)

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula a função de decisão f(x) usando apenas os m landmarks.

        f(x) = K(x, landmarks) · (W⁻¹ Cᵀ α) + b

        Memória O(n_test × m) — consistente com a aproximação usada no treino.
        """
        if self.alpha_ is None:
            raise ValueError("Modelo não foi treinado. Chame fit() primeiro.")

        K_test_lm = self.kernel(X, self.nystrom_.landmarks_)   # n_test × m
        return K_test_lm @ self._w_alpha_ + self.b_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prediz os rótulos.

        Parâmetros
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Dados de teste

        Retorna
        -------
        y_pred : np.ndarray, shape (n_samples,)
            Rótulos preditos (+1 ou -1)
        """
        return np.sign(self.decision_function(X))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcula a acurácia.

        Parâmetros
        ----------
        X : np.ndarray
            Dados de teste
        y : np.ndarray
            Rótulos verdadeiros

        Retorna
        -------
        accuracy : float
            Fração de predições corretas
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


if __name__ == "__main__":
    # Exemplo de uso
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Gerar dados
    X, y = make_classification(n_samples=500, n_features=10,
                               n_informative=5, random_state=42)
    y = 2 * y - 1  # Converter para {-1, +1}

    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Normalizar
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Teste da Aproximação de Nyström e LSSVM")
    print("=" * 50)
    print(f"Treino: {X_train.shape[0]} amostras")
    print(f"Teste: {X_test.shape[0]} amostras")
    print()

    # Testar aproximação de Nyström
    print("1. Teste de aproximação de Nyström")
    print("-" * 40)

    kernel = RBFKernel(sigma=1.0)
    K_true = kernel(X_train)

    for method in ['random', 'kmeans', 'opposite']:
        for m_ratio in [0.1, 0.2, 0.5]:
            m = int(m_ratio * X_train.shape[0])
            nystrom = NystromApproximation(
                n_landmarks=m,
                kernel=kernel,
                selection_method=method,
                random_state=42
            )
            nystrom.fit(X_train)

            frob_err, spec_err = nystrom.approximation_error(K_true)
            print(f"{method:10s} m={m:3d} ({m_ratio*100:4.0f}%): "
                  f"Frobenius={frob_err:.4f}, Spectral={spec_err:.4f}")

    print()

    # Testar LSSVM com Nyström
    print("2. Teste de LSSVM com Nyström")
    print("-" * 40)

    for method in ['random', 'kmeans', 'opposite']:
        m = int(0.2 * X_train.shape[0])
        model = NystromLSSVM(
            n_landmarks=m,
            gamma=1.0,
            kernel=RBFKernel(sigma=1.0),
            selection_method=method,
            random_state=42
        )
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        print(f"{method:10s}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
