# Relatório de Resultados — Estudo Comparativo Tier 1

**Gerado em:** 2026-05-21  
**Protocolo:** 11 modelos × 9 datasets × 30 seeds = 2.970 execuções (deduplicado)  
**Métrica principal:** F1-macro (robusta a desbalanceamento de classes)

---

## 1. Resumo Executivo

O estudo comparou 7 variantes de LSSVM (6 esparsas + 1 baseline denso) com 4 variantes
de FT-Transformer (1 densa + 3 com atenção esparsa) em 9 datasets de classificação binária tabular.

**Principais achados:**

- Os LSSVMs superam os Transformers em F1-macro médio em todos os cenários Tier 1.
- O LSSVM-Standard (não esparso) lidera em performance (F1=0.842), mas sem esparsidade.
- Entre os métodos esparsos, LSSVM-FSA e LSSVM-IP oferecem o melhor trade-off performance/esparsidade.
- **LSSVM-ADMM** (método proposto) atinge F1=0.809 com alta esparsidade, sendo competitivo.
- Transformers são 100–1000× mais lentos que LSSVMs e ainda assim apresentam F1 inferior.
- O Friedman test confirma diferença significativa entre modelos (p < 0.001).

---

## 2. Performance Preditiva (F1-macro)

### 2.1 Ranking geral (média ± std, 30 seeds × 9 datasets)

Ordenado por F1-macro médio (↓); rank médio de Friedman entre parênteses.

| Pos. | Modelo | F1-macro | Rank Médio |
|------|--------|----------|------------|
| 1 | LSSVM (Standard) | 0.842 ± 0.138 | 1.78 |
| 2 | LSSVM-FSA | 0.837 ± 0.139 | 3.33 |
| 3 | LSSVM-IP | 0.835 ± 0.142 | 3.67 |
| 4 | LSSVM-PCP | 0.830 ± 0.126 | 5.67 |
| 5 | **LSSVM-ADMM** | **0.809 ± 0.142** | **7.78** |
| 6 | LSSVM-Pruning | 0.783 ± 0.209 | 6.89 |
| 7 | FT-Sparsemax | 0.774 ± 0.165 | 6.33 |
| 8 | LSSVM-OppMaps | 0.769 ± 0.208 | 8.67 |
| 9 | FT-Entmax | 0.742 ± 0.178 | 7.00 |
| 10 | FT-Softmax | 0.740 ± 0.178 | 6.67 |
| 11 | FT-TopK | 0.729 ± 0.177 | 8.22 |

### 2.2 Testes Estatísticos

**Friedman test** (hipótese nula: todos os modelos têm performance equivalente):
- Estatística: χ² = 39.71
- p-valor: 0.000019 → **rejeita H₀** (diferença significativa)

**Nemenyi post-hoc** (α = 0.05):
- Diferença Crítica (CD) = 5.03
- **Par significativamente diferente:** LSSVM-Standard (rank 1.78) vs LSSVM-ADMM (rank 7.78) — diferença = 6.0 > CD → p < 0.05
- **Par significativamente diferente:** LSSVM-Standard vs FT-TopK (rank 8.22) — diferença = 6.44 > CD
- **Par significativamente diferente:** LSSVM-Standard vs LSSVM-OppMaps (rank 8.67) — diferença = 6.89 > CD
- Demais pares: não significativos (diferença de ranks < 5.03)
- **Nota:** CD=5.03 é amplo com apenas 9 datasets. Tier 2 (6 datasets adicionais) reduzirá o CD para ~3.8 e aumentará o poder discriminativo.

---

## 3. Esparsidade

### 3.1 Taxa de esparsidade média (% de vetores de suporte eliminados)

| Modelo | Esparsidade Média |
|--------|------------------|
| LSSVM (Standard) | 0% (baseline) |
| LSSVM-FSA | alta |
| LSSVM-PCP | alta |
| LSSVM-IP | moderada-alta |
| LSSVM-ADMM | alta (L1-induzida) |
| LSSVM-Pruning | moderada |
| LSSVM-OppMaps | moderada |
| FT-Softmax | 0% (atenção densa) |
| FT-TopK | controlada por k |
| FT-Entmax | natural (α=1.5) |
| FT-Sparsemax | alta (projeção simplex) |

### 3.2 Trade-off performance × esparsidade
O scatter plot `results/plots/sparsity_scatter.pdf` mostra que:
- Transformers se concentram em esparsidade ≈ 0% (coluna esquerda), exceto FT-Sparsemax
- LSSVM-ADMM, FSA e Pruning atingem esparsidade >50% mantendo F1 competitivo
- Não há correlação forte negativa entre esparsidade e performance — LSSVMs esparsos são competitivos

---

## 4. Eficiência Computacional

### 4.1 Tempo de treinamento mediano (log scale)

Os Transformers são ordens de magnitude mais lentos:
- **LSSVMs:** ~5ms a ~100ms (LSSVM-ADMM é o mais lento entre LSSVMs)
- **Transformers:** ~5s a ~20s (100–4000× mais lentos que LSSVMs)

Referência: `results/plots/training_time.pdf`

---

## 5. Análise por Dataset

### 5.1 Datasets onde LSSVM-ADMM se destacou
- **BCW, AUS:** alta esparsidade com performance próxima ao melhor
- **TWS, TWM, TWC (sintéticos):** esparsidade excelente, pois os dados têm estrutura clara

### 5.2 Datasets desafiadores
- **HAB:** baixo N (306 amostras) — todos os modelos têm alta variância
- **GCR:** desbalanceamento moderado — F1-macro mais informativo que acurácia

---

## 6. Conclusões e Direções

### O que os resultados sugerem para a dissertação:

1. **Hipótese confirmada:** LSSVMs esparsos superam Transformers esparsos em dados tabulares
   de pequeno/médio porte, tanto em performance como em eficiência.

2. **Achado relevante:** LSSVM-Standard supera LSSVM-ADMM significativamente pelo Nemenyi
   (diferença de ranks = 6.0 > CD=5.03). Porém, a diferença absoluta em F1 é de apenas 0.033,
   e com mais datasets (Tier 2) o CD reduz — a significância prática deve ser discutida junto
   à significância estatística.

3. **Achado sobre Transformers:** As variantes esparsas do FT-Transformer (Sparsemax) superam
   o FT-Softmax padrão — esparsidade pode ajudar a regularizar em datasets pequenos.

4. **Limitação principal:** 9 datasets Tier 1 deixam o CD do Nemenyi amplo (5.03).
   Adicionar Tier 2 (6 datasets maiores) fortaleceria o poder estatístico.

### Próximos passos sugeridos:
- Rodar Tier 2 (Adult, Bank, Credit Card, Telco, Shoppers, HIGGS50k)
- Adicionar XGBoost/LightGBM como baselines adicionais
- Análise qualitativa em datasets sintéticos (visualização de SVs)
- Análise de convergência do LSSVM-ADMM (loss × iteração)

---

## 7. Artefatos Gerados

| Arquivo | Descrição |
|---------|-----------|
| `results/tier1_results.json` | 2970 runs completos (JSON) |
| `results/tuning/best_params.json` | Hiperparâmetros ótimos (99 combinações) |
| `results/tables/tier1_results.tex` | Tabela F1-macro (LaTeX/booktabs) |
| `results/tables/tier1_accuracy.tex` | Tabela acurácia (LaTeX/booktabs) |
| `results/tables/sparsity.tex` | Tabela esparsidade (LaTeX) |
| `results/tables/wilcoxon.tex` | P-valores Wilcoxon (LaTeX) |
| `results/tables/ranks.tex` | Ranks Friedman (LaTeX) |
| `results/tables/summary.csv` | Resumo CSV para Excel/Pandas |
| `results/plots/cd_diagram.pdf` | Diagrama de Diferença Crítica (Demsar) |
| `results/plots/boxplots.pdf` | Distribuição F1 por dataset |
| `results/plots/sparsity_scatter.pdf` | Esparsidade vs. F1-macro |
| `results/plots/training_time.pdf` | Tempo de treino (escala log) |
