# Síntese dos Resultados — Tier 1

## Configuração experimental
- **Modelos:** 11 (7 LSSVM + 4 FT-Transformer)
- **Datasets:** 9 (6 reais + 3 sintéticos)
- **Seeds:** 30 por combinação
- **Total de runs:** 2.970
- **Métrica principal:** F1-macro
- **Tuning:** Optuna TPE, 100 trials (LSSVM) / 30 trials (Transformer), 5-fold CV

## Ranking de performance (F1-macro médio)

1. LSSVM (Standard) — 0.842
2. LSSVM-FSA — 0.837
3. LSSVM-IP — 0.835
4. LSSVM-PCP — 0.830
5. **LSSVM-ADMM — 0.809** ← método proposto
6. LSSVM-Pruning — 0.783
7. FT-Sparsemax — 0.774
8. LSSVM-OppMaps — 0.769
9. FT-Softmax — 0.742
10. FT-Entmax — 0.742
11. FT-TopK — 0.729

## Teste de Friedman
- χ² = 39.71, p = 0.000019 → diferença significativa entre modelos

## Nemenyi CD (α = 0.05)
- CD = 5.03 (com 9 datasets)
- Pares significativamente diferentes (|rank_i − rank_j| > CD):
  - LSSVM-Standard (1.78) vs LSSVM-ADMM (7.78): diff = 6.0 → **significativo**
  - LSSVM-Standard (1.78) vs FT-TopK (8.22): diff = 6.44 → **significativo**
  - LSSVM-Standard (1.78) vs LSSVM-OppMaps (8.67): diff = 6.89 → **significativo**
- Todos os demais pares: não significativos
- Mais datasets (Tier 2) reduzirão o CD para ~3.8 e aumentarão o poder discriminativo

## Conclusão central
LSSVMs superam FT-Transformers em performance, velocidade e esparsidade em datasets
tabulares de pequeno/médio porte. LSSVM-Standard supera LSSVM-ADMM significativamente
pelo Nemenyi, mas a diferença absoluta em F1 é apenas 0.033 — o ADMM oferece esparsidade
real (30% de redução média de SVs) com custo de performance limitado.
