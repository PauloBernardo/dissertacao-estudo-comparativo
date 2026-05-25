# Sparse LSSVM vs Sparse Tabular Transformers — Estudo Comparativo

Dissertação de mestrado: estudo experimental comparando métodos de esparsidade em
**Least Squares SVM (LSSVM)** e **Transformers tabulares** para classificação binária tabular.

Estende o trabalho de Marinho et al. (2023): *"Sparse Least Square SVM in Primal via Nesterov
Accelerated Alternating Directions Method of Multipliers"*, IWANN 2023.

---

## Modelos Comparados (11)

### LSSVM
| Modelo | Tipo | Referência |
|--------|------|-----------|
| LSSVM-Standard | Baseline (não esparso) | Suykens & Vandewalle, 1999 |
| LSSVM-ADMM | Primal + L1 + ADMM-Nesterov | **Marinho et al., 2023** |
| LSSVM-PCP | Primal + Pivoted Cholesky | Zhou, 2015 |
| LSSVM-FSA | Primal + Forward Selection | Jiao et al., 2007 |
| LSSVM-Pruning | Dual + Pruning iterativo | Suykens et al. |
| LSSVM-IP | Dual + Identificação de protótipos | Carvalho & Braga, 2009 |
| LSSVM-OppMaps | Dual + Opposite Maps | Neto & Barreto, 2013 |

### FT-Transformer (Gorishniy et al., 2021) com variantes de atenção
| Modelo | Atenção |
|--------|---------|
| FT-Softmax | Softmax padrão (denso) |
| FT-TopK | Top-k esparso |
| FT-Entmax | α-Entmax (α=1.5) |
| FT-Sparsemax | Sparsemax |

---

## Datasets (Tier 1 — 9 datasets)

| Dataset | N | Features | Fonte |
|---------|---|----------|-------|
| BCW | 569 | 30 | UCI |
| PID | 768 | 8 | UCI |
| HAB | 306 | 3 | UCI |
| VCP | 310 | 6 | UCI |
| GCR | 1000 | 20 | UCI |
| AUS | 690 | 14 | UCI |
| TWS | 400 | 2 | Sintético |
| TWM | 400 | 2 | Sintético |
| TWC | 400 | 2 | Sintético |

---

## Resultados Principais (Tier 1 — F1-macro, 30 seeds)

Ordenado por F1-macro médio (30 seeds × 9 datasets).

| Modelo | F1-macro | Rank Médio (Friedman) |
|--------|----------|-----------------------|
| LSSVM (Standard) | 0.842 ± 0.138 | 1.78 |
| LSSVM-FSA | 0.837 ± 0.139 | 3.33 |
| LSSVM-IP | 0.835 ± 0.142 | 3.67 |
| LSSVM-PCP | 0.830 ± 0.126 | 5.67 |
| **LSSVM-ADMM** | **0.809 ± 0.142** | **7.78** |
| LSSVM-Pruning | 0.783 ± 0.209 | 6.89 |
| FT-Sparsemax | 0.774 ± 0.165 | 6.33 |
| LSSVM-OppMaps | 0.769 ± 0.208 | 8.67 |
| FT-Entmax | 0.742 ± 0.178 | 7.00 |
| FT-Softmax | 0.740 ± 0.178 | 6.67 |
| FT-TopK | 0.729 ± 0.177 | 8.22 |

**Friedman test:** χ² = 39.71, p < 0.001 — diferença significativa entre modelos.
**Nemenyi CD (α=0.05):** 5.03 — Standard significativamente superior a ADMM, FT-TopK e OppMaps.

---

## Estrutura do Repositório

```
├── config/              # YAML: datasets, modelos, search spaces
├── src/
│   ├── models/          # implementações LSSVM e Transformers
│   ├── metrics/         # performance, esparsidade, eficiência, testes estatísticos
│   ├── tuning/          # Bayesian optimization via Optuna
│   ├── experiments/     # runner principal
│   ├── data/            # loaders e pré-processamento
│   └── analysis/        # tabelas LaTeX e figuras
├── scripts/
│   ├── run_tuning_tier1.py      # tuning (Optuna, 5-fold CV)
│   ├── run_experiments_tier1.py # experimentos (30 seeds)
│   └── generate_analysis.py    # tabelas + figuras
├── tests/               # 124 testes (pytest)
└── results/
    ├── tier1_results.json       # 2970 runs (11 × 9 × 30)
    ├── tables/                  # LaTeX prontos para a dissertação
    └── plots/                   # PDF e PNG
```

---

## Como Reproduzir

### 1. Ambiente
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Testes
```bash
pytest tests/ -q
# 124 passed, 5 skipped (network)
```

### 3. Tuning (Tier 1)
```bash
python scripts/run_tuning_tier1.py \
    --trials-lssvm 100 --trials-transformer 30 \
    --folds 5 --timeout-transformer 900
# Resumível: pula combinações já salvas em results/tuning/best_params.json
```

### 4. Experimentos (Tier 1)
```bash
python scripts/run_experiments_tier1.py --seeds 30
# Resumível: pula runs já salvos em results/tier1_results.json
```

### 5. Análise
```bash
python scripts/generate_analysis.py
# Gera results/tables/*.tex e results/plots/*.pdf
```

---

## Referências Principais

- Marinho et al. (2023). *Sparse Least Square SVM in Primal via Nesterov Accelerated ADMM*. IWANN.
- Gorishniy et al. (2021). *Revisiting Deep Learning Models for Tabular Data*. NeurIPS.
- Suykens & Vandewalle (1999). *Least Squares Support Vector Machine Classifiers*. Neural Processing Letters.
- Demšar (2006). *Statistical Comparisons of Classifiers over Multiple Data Sets*. JMLR.
- Peters et al. (2019). *Sparse Sequence-to-Sequence Models*. ACL.
- Martins & Astudillo (2016). *From Softmax to Sparsemax*. ICML.
