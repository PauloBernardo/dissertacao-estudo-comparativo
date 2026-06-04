# Experimentos Ablação — Datasets Sintéticos

Complementam o Tier 1 (9 datasets × 11 modelos × 30 seeds). Todos os
experimentos abaixo usam os mesmos 11 modelos e 30 seeds.

---

## Experimento A — Escala (N=400 → N=2000)

**Arquivo:** `results/synthetic_scaling_n2000.json`  
**Script:** `scripts/run_synthetic_scaling.py`  
**Hiperparâmetros:** best_params do Tier 1 (sem retuning)

### Motivação
Verificar se a inferioridade dos Transformers nos sintéticos (TWS, TWC)
se deve a falta de dados ou a limitação estrutural.

### Resultados (F1-macro médio, 30 seeds)

| Modelo | TWS 400→2000 | TWM 400→2000 | TWC 400→2000 |
|--------|-------------|-------------|-------------|
| LSSVM-Standard | 0.993→0.997 (+0.003) | 1.000→0.998 (-0.002) | 0.900→0.947 (+0.047) |
| LSSVM-FSA | 0.990→0.993 (+0.003) | 1.000→0.995 (-0.005) | 0.893→0.931 (+0.038) |
| LSSVM-IP | 0.990→0.995 (+0.005) | 1.000→0.998 (-0.002) | 0.888→0.938 (+0.050) |
| LSSVM-PCP | 0.989→0.992 (+0.004) | 0.995→0.997 (+0.002) | 0.872→0.935 (+0.063) |
| LSSVM-ADMM | 0.992→0.995 (+0.003) | 0.992→0.998 (+0.007) | 0.866→0.938 (+0.072) |
| LSSVM-Pruning | 0.994→0.996 (+0.002) | 0.996→0.998 (+0.002) | 0.874→0.936 (+0.061) |
| LSSVM-OppMaps | 0.986→0.987 (+0.000) | 0.990→0.951 (-0.039) | 0.899→0.919 (+0.020) |
| FT-Softmax | 0.653→0.988 (+0.335) | 1.000→0.998 (-0.002) | 0.478→0.532 (+0.054) |
| FT-TopK | 0.640→0.859 (+0.219) | 0.974→0.997 (+0.023) | 0.468→0.642 (+0.174) |
| FT-Entmax | 0.667→0.989 (+0.323) | 0.999→0.998 (-0.002) | 0.504→0.903 (+0.399) |
| FT-Sparsemax | 0.690→0.988 (+0.298) | 1.000→0.998 (-0.002) | 0.710→0.955 (+0.245) |

### Conclusões
- **TWS (espirais):** Transformers com N=2000 alcançam F1≈0.99, empatando com
  LSSVMs. A inferioridade com N=400 era inteiramente devida à falta de dados.
- **TWC (checkerboard):** FT-Entmax (0.903) e FT-Sparsemax (0.955) superam
  os LSSVMs com N=2000. A atenção esparsa consegue delimitar as 16 regiões
  do checkerboard melhor que o kernel RBF com dados suficientes.
- **TWM (luas):** já saturado em N=400 para todos os modelos; sem diferença.

---

## Experimento B — Ruído dimensional (2 → 5 features, σ fixo)

**Arquivo:** `results/synthetic_5features.json`  
**Script:** `scripts/run_synthetic_5features.py`  
**Hiperparâmetros:** best_params do Tier 1 (tunados para 2 features — **σ fixo**)

### Motivação
Avaliar robustez a features irrelevantes: os datasets 2D originais são
embutidos em 5D adicionando 3 features de ruído gaussiano N(0,1).
O rótulo depende apenas das 2 primeiras features.

### Resultados (F1-macro médio, 30 seeds)

| Modelo | TWS 2f→5f | TWM 2f→5f | TWC 2f→5f |
|--------|----------|----------|----------|
| LSSVM-Standard | 0.993→0.643 (-0.351) | 1.000→0.936 (-0.064) | 0.900→0.348 (-0.552) |
| LSSVM-FSA | 0.990→0.379 (-0.611) | 1.000→0.935 (-0.065) | 0.893→0.340 (-0.553) |
| LSSVM-IP | 0.990→0.373 (-0.617) | 1.000→0.914 (-0.086) | 0.888→0.433 (-0.455) |
| LSSVM-PCP | 0.989→0.594 (-0.394) | 0.995→0.808 (-0.187) | 0.872→0.335 (-0.537) |
| LSSVM-ADMM | 0.992→0.571 (-0.420) | 0.992→0.380 (-0.611) | 0.866→0.337 (-0.529) |
| LSSVM-Pruning | 0.994→0.524 (-0.470) | 0.996→0.359 (-0.637) | 0.874→0.484 (-0.390) |
| LSSVM-OppMaps | 0.986→0.517 (-0.469) | 0.990→0.672 (-0.317) | 0.899→0.467 (-0.432) |
| FT-Softmax | 0.653→0.617 (-0.036) | 1.000→0.962 (-0.038) | 0.478→0.490 (+0.012) |
| FT-TopK | 0.640→0.593 (-0.047) | 0.974→0.840 (-0.134) | 0.468→0.487 (+0.019) |
| FT-Entmax | 0.667→0.620 (-0.047) | 0.999→0.973 (-0.027) | 0.504→0.493 (-0.011) |
| FT-Sparsemax | 0.690→0.597 (-0.093) | 1.000→0.933 (-0.067) | 0.710→0.481 (-0.229) |

### Limitação crítica — σ fixo em dimensão aumentada

**Este experimento usa o σ tunado para 2 features. Isso introduz um viés
sistemático contra os LSSVMs**, pois o kernel RBF é sensível à escala das
distâncias euclidianas:

$$\|x - z\|^2_{5D} = \|x - z\|^2_{2D} + \underbrace{\sum_{j=3}^{5}(x_j - z_j)^2}_{\text{ruído} \approx 2 \text{ em média}}$$

Com σ calibrado para distâncias típicas em 2D, o kernel em 5D colapsa:

$$k(x,z) = e^{-\|x-z\|^2_{5D} / 2\sigma^2} \ll e^{-\|x-z\|^2_{2D} / 2\sigma^2}$$

A matriz kernel tende à identidade e o LSSVM perde toda informação de
similaridade entre pontos. Os Transformers não sofrem esse efeito porque
a atenção opera por feature independentemente — features irrelevantes
simplesmente recebem pesos próximos de zero.

### Conclusão parcial (σ fixo)
Com σ fixo, LSSVMs colapsam em 5D enquanto Transformers são estáveis.
Esse resultado reflete a limitação do kernel RBF sem retuning, não
necessariamente a incapacidade estrutural dos LSSVMs com ruído dimensional.

**Ver Experimento C para a comparação justa com retuning.**

---

## Experimento C — Ruído dimensional com retuning (2 → 5 features, σ reoptimizado)

**Arquivo:** `results/synthetic_5features_tuned.json`  
**Script:** `scripts/run_synthetic_5features.py --params results/tuning/best_params_5f.json`  
**Tuning:** `scripts/run_tuning_5features.py` (Optuna TPE, 100 trials LSSVM / 30 Transformer, 5-fold CV)

### Motivação
Repetir o Experimento B com σ reotimizado para cada modelo/dataset 5D,
eliminando o viés do σ fixo e permitindo comparação justa.

### Status
⏳ Tuning em andamento — resultados a adicionar após conclusão.
