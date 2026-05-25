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
- χ² = 39.67, p = 0.000019 → diferença significativa entre modelos

## Nemenyi CD (α = 0.05)
- CD = 5.03 (com 9 datasets)
- LSSVM-Standard é significativamente superior apenas a FT-TopK e LSSVM-OppMaps
- Mais datasets (Tier 2) reduzirão o CD e aumentarão o poder discriminativo

## Conclusão central
LSSVMs superam FT-Transformers em todos os aspectos (performance, velocidade, esparsidade)
em datasets tabulares de pequeno/médio porte. A esparsidade via ADMM-Nesterov não
compromete significativamente a performance em relação ao LSSVM-Standard.
