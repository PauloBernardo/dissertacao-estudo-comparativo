#!/usr/bin/env python3
"""Auditoria matemática do NystromLSSVM corrigido.

Verifica três propriedades fundamentais:

1. CORRETUDE MATEMÁTICA
   f(x) = K(x, Z) · wα + b
   deve ser EXATAMENTE igual a
   f(x) = K(x, Z) W⁻¹ Cᵀ α + b   (Nyström approximation completa)
   E DIFERENTE de
   f(x) = K(x, X_train) α + b    (BUG: kernel completo)

2. SEM VAZAMENTO DE X_train
   Depois de treinar, podemos APAGAR X_train_ e a predição
   continua funcionando, porque só dependa de landmarks Z.

3. CUSTO O(n_test × m)
   Tempo de predição escala com m, não com n.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.models.nystrom import NystromLSSVM, RBFKernel
from src.models.nystrom_lssvm_wrapper import NystromLSSVMColnorm


def banner(title):
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print(f"{'═'*70}")


# ───────────────────────────────────────────────────────────────────────────
# Setup: dataset razoável (n grande para diferenciar m vs n claramente)
# ───────────────────────────────────────────────────────────────────────────
banner("Setup")
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                           random_state=42)
y = 2 * y - 1  # → {-1, +1}
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=42)

sc = StandardScaler().fit(X_tr)
X_tr = sc.transform(X_tr)
X_te = sc.transform(X_te)

print(f"X_train: {X_tr.shape}   X_test: {X_te.shape}")

model = NystromLSSVMColnorm(sigma=2.0, gamma=10.0, m_ratio=0.10, random_state=0)
model.fit(X_tr, y_tr)

n = X_tr.shape[0]
m = model.n_support_
print(f"n = {n},  m = {m},  m/n = {m/n:.1%},  esparsidade = {1-m/n:.1%}")


# ───────────────────────────────────────────────────────────────────────────
# CHECK 1: Equivalência matemática das duas fórmulas
# ───────────────────────────────────────────────────────────────────────────
banner("CHECK 1 — Corretude matemática da predição")

ker  = RBFKernel(sigma=2.0)
inner = model._model
Z    = inner.nystrom_.landmarks_       # (m, d)
W    = ker(Z, Z)                       # (m, m)
W_inv = inner.nystrom_.W_inv_          # (m, m)
C    = inner.nystrom_.C_               # (n, m)  = K(X_train, Z)
alpha = inner.alpha_                   # (n,)
b    = inner.b_

K_te_lm = ker(X_te, Z)                 # (n_test, m)
K_te_tr = ker(X_te, X_tr)              # (n_test, n)  — só para comparar

# F1: fórmula do código (cache w_α)
f1 = K_te_lm @ inner._w_alpha_ + b

# F2: fórmula expandida — Nyström aproximação completa
#     f(x) = K(x, Z) · W⁻¹ · Cᵀ · α + b
w_alpha_explicit = W_inv @ (C.T @ alpha)
f2 = K_te_lm @ w_alpha_explicit + b

# F3: BUG — kernel completo K(x, X_train) · α
f3 = K_te_tr @ alpha + b

# F4: BUG — α subsetado nos landmarks
lm_idx = inner.nystrom_.selector_.indices_
alpha_subset = alpha[lm_idx]
f4 = K_te_lm @ alpha_subset + b

print(f"||F1 (cache w_α)   - F2 (W⁻¹Cᵀα explícito)||  = {np.linalg.norm(f1 - f2):.2e}")
print(f"   → Devem ser idênticas (mesma fórmula matemática).")

print(f"||F1 (correto)     - F3 (BUG: K_full α)||     = {np.linalg.norm(f1 - f3):.2e}")
print(f"   → Devem ser DIFERENTES (kernel completo seria vazamento).")

print(f"||F1 (correto)     - F4 (BUG: α[lm_idx])||    = {np.linalg.norm(f1 - f4):.2e}")
print(f"   → Devem ser DIFERENTES (subset não recupera Nyström).")

# Acurácia de cada formulação
acc = lambda f: np.mean(np.sign(f) == y_te)
print(f"\nAcurácia:")
print(f"  F1 (código corrigido)            : {acc(f1):.4f}")
print(f"  F2 (Nyström W⁻¹Cᵀα explícito)    : {acc(f2):.4f}")
print(f"  F3 (BUG: kernel completo)        : {acc(f3):.4f}")
print(f"  F4 (BUG: subset α em landmarks)  : {acc(f4):.4f}")

print()
if np.allclose(f1, f2, atol=1e-10):
    print("✓ F1 == F2: fórmula do código é matematicamente a aproximação Nyström.")
else:
    print("✗ ERRO: F1 ≠ F2 — código diverge da definição matemática!")

if not np.allclose(f1, f3, atol=1e-3):
    print("✓ F1 ≠ F3: código NÃO faz a versão buggy do kernel completo.")
else:
    print("✗ ERRO: F1 ≈ F3 — código pode estar usando kernel completo!")


# ───────────────────────────────────────────────────────────────────────────
# CHECK 2: Sem vazamento de X_train
# ───────────────────────────────────────────────────────────────────────────
banner("CHECK 2 — Predição NÃO depende de X_train (só de landmarks Z)")

f_before = model._model.decision_function(X_te)
print(f"f(X_test) antes de apagar X_train_  : {f_before[:5]}")

# Apaga X_train do modelo
X_train_backup = inner.X_train_
inner.X_train_ = None  # vazamento → erro

try:
    f_after = inner.decision_function(X_te)
    diff = np.linalg.norm(f_before - f_after)
    print(f"f(X_test) DEPOIS de apagar X_train_ : {f_after[:5]}")
    print(f"||diferença|| = {diff:.2e}")
    if diff < 1e-12:
        print("✓ Predição idêntica → modelo não usa X_train, só os landmarks.")
    else:
        print("✗ ERRO: predição mudou → modelo está usando X_train escondido.")
except Exception as e:
    print(f"✗ ERRO: predição quebrou após apagar X_train_: {e}")
finally:
    inner.X_train_ = X_train_backup


# ───────────────────────────────────────────────────────────────────────────
# CHECK 3: Custo O(n_test × m), não O(n_test × n)
# ───────────────────────────────────────────────────────────────────────────
banner("CHECK 3 — Custo de predição escala com m, não com n")

# Dataset MAIOR para destacar n vs m
X_big, y_big = make_classification(n_samples=8000, n_features=20,
                                    n_informative=10, random_state=42)
y_big = 2 * y_big - 1
X_bt, X_bte, y_bt, y_bte = train_test_split(X_big, y_big, test_size=0.25,
                                             random_state=42)
sc2 = StandardScaler().fit(X_bt)
X_bt  = sc2.transform(X_bt)
X_bte = sc2.transform(X_bte)

# Mesmo m absoluto em três tamanhos de X_train
for n_use, m_use in [(1000, 100), (3000, 100), (6000, 100)]:
    Xt = X_bt[:n_use]
    yt = y_bt[:n_use]
    mdl = NystromLSSVMColnorm(sigma=2.0, gamma=10.0,
                              m_ratio=m_use/n_use, random_state=0)
    mdl.fit(Xt, yt)
    actual_m = mdl.n_support_

    # Tempo de predição
    t0 = time.perf_counter()
    for _ in range(10):
        _ = mdl._model.decision_function(X_bte)
    t_pred = (time.perf_counter() - t0) / 10

    # Memória da matriz K_test_lm
    mem_lm  = X_bte.shape[0] * actual_m * 8  # n_test × m × 8 bytes (float64)
    mem_full = X_bte.shape[0] * n_use   * 8  # se fosse n_test × n
    print(f"n_train={n_use:5d}  m={actual_m:3d}   "
          f"predict={t_pred*1000:6.1f}ms   "
          f"K_test_lm={mem_lm/1024:.0f}KB  (vs K_test_full={mem_full/1024:.0f}KB)")

print("\nSe o tempo NÃO cresce com n_train (mantendo m fixo) → custo O(n_test×m). ✓")


# ───────────────────────────────────────────────────────────────────────────
# CHECK 4: Comparação direta — versão honesta vs bug emulado
# ───────────────────────────────────────────────────────────────────────────
banner("CHECK 4 — F1 do BCW: implementação atual vs simulação dos bugs")

from src.data.loaders import DatasetLoader
from sklearn.metrics import f1_score

X_bcw, y_bcw, _ = DatasetLoader.load("BCW")
y_signed = (y_bcw * 2 - 1).astype(int)

f1_correct, f1_bug_full, f1_bug_subset = [], [], []
for seed in range(10):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_bcw, y_signed, test_size=0.30, stratify=y_signed, random_state=seed)
    sc = StandardScaler().fit(X_tr)
    X_tr_p = sc.transform(X_tr)
    X_te_p = sc.transform(X_te)

    # Hyperparams tuned para BCW
    mdl = NystromLSSVMColnorm(sigma=7.096, gamma=6.791, m_ratio=0.204,
                              random_state=seed)
    mdl.fit(X_tr_p, y_tr)
    inner = mdl._model

    # F1 correto (código atual)
    p_correct = np.sign(inner.decision_function(X_te_p))

    # F1 com bug: kernel completo
    ker = RBFKernel(sigma=7.096)
    p_bug_full = np.sign(ker(X_te_p, X_tr_p) @ inner.alpha_ + inner.b_)

    # F1 com bug: subset α
    lm_idx = inner.nystrom_.selector_.indices_
    Z = inner.nystrom_.landmarks_
    p_bug_subset = np.sign(ker(X_te_p, Z) @ inner.alpha_[lm_idx] + inner.b_)

    def f1m(pred):
        yb = ((y_te + 1) // 2).astype(int)
        pb = ((pred + 1) // 2).astype(int)
        return f1_score(yb, pb, average="macro", zero_division=0)

    f1_correct.append(f1m(p_correct))
    f1_bug_full.append(f1m(p_bug_full))
    f1_bug_subset.append(f1m(p_bug_subset))

print(f"BCW F1-macro (10 seeds, params tuned para σ=7.096, γ=6.791, m/n=0.204):")
print(f"  Implementação atual (W⁻¹Cᵀα)    : {np.mean(f1_correct):.4f} ± {np.std(f1_correct):.4f}")
print(f"  BUG: kernel completo (kernel inflado): {np.mean(f1_bug_full):.4f} ± {np.std(f1_bug_full):.4f}")
print(f"  BUG: α[lm_idx]                  : {np.mean(f1_bug_subset):.4f} ± {np.std(f1_bug_subset):.4f}")
print()
print("→ Se o atual ≈ bug-completo: estamos com vazamento.")
print("→ Se o atual << bug-subset: simulando uma 'fix' ingênua daria 0.668.")
