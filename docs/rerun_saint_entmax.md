# Re-execução SAINT (fiel) + FT-Entmax (corrigido) — procedimento

Origem: auditoria de 2026-09-18 (`docs/model_references.md`, rodada 2026-09-18).

## O que mudou no código
| Item | Antes | Depois |
|---|---|---|
| `sparse_attention/entmax_attention.py` | intervalo de bisseção errado → saída não era entmax-α (Σp≈1,4–7 antes de renormalizar) | intervalo de Peters et al. 2019 + `autograd.Function` com Jacobiana exata; testes `TestEntmaxBisectionRegression` |
| `ft_transformer_model.py::SAINTClassifier` | tokenizador linear do FT + inter-instâncias só no CLS | arquitetura do artigo: embedding FC+ReLU por atributo, MSA→FF→MISA→FF com LN nos resíduos e GELU, MISA sobre (p+1)·d, head MLP no CLS; versão antiga preservada como `SAINTClassifierCLSOnly`; testes `TestSAINTFaithful` |
| `landmark_selection.py` | `get_selector('opposite')` = reflexão OBL rotulada como Opposite Maps | chave `'obl_reflection'`; `'opposite'` levanta erro (fiel: `select_opposite_landmarks`) |

## Achado adicional (Ablação A)
`results/ablation_a_transformers.json` foi gerado por `run_tier1_gridcv.py` nos datasets `*_2k`,
isto é, com **GridSearchCV re-executado em N=2000**, enquanto os LSSVMs/XGBoost usaram
`run_ablation_a_scaling.py` (transferência dos hiperparâmetros do Tier 1, `protocol: transfer_from_tier1`).
A dissertação afirma transferência para todos. O notebook de re-execução corrige isso rodando
**os seis Transformers** na Ablação A com `run_ablation_a_scaling.py --transformers-only`
(360 ajustes, sem CV — barato). Após o merge, o texto da Ablação A fica verdadeiro, mas os
números de TODOS os Transformers nessa ablação mudam.

## Passo a passo
1. **Commit + push** de `src/`, `tests/`, `scripts/merge_rerun_results.py`, `notebooks/saint_entmax_rerun_kaggle.ipynb` (o notebook clona o GitHub).
2. Rodar `notebooks/saint_entmax_rerun_kaggle.ipynb` no Kaggle (GPU T4). Saídas em `/kaggle/working`:
   `rerun_saint_entmax_{tier1,tier2,n5000,ablA,ablBC,scaling}.json`.
3. Baixar os JSONs para `results/` e fazer o merge (cada comando faz backup `*_pre_saintentmax_backup.json`):
   ```bash
   V="SAINTColnorm FTTransformer_entmax"; TAG=saintentmax
   python scripts/merge_rerun_results.py --target results/tier1_gridcv.json        --source results/rerun_saint_entmax_tier1.json --variants $V --tag $TAG
   python scripts/merge_rerun_results.py --target results/tier2_transformers.json  --source results/rerun_saint_entmax_tier2.json --variants $V --tag $TAG
   python scripts/merge_rerun_results.py --target results/tier2_fixedparams_n5000_transformers.json --source results/rerun_saint_entmax_n5000.json --variants $V --tag $TAG
   # Ablação B e C: o rerun contém os dois; separar por dataset
   python - <<'PY'
   import json
   r=json.load(open('results/rerun_saint_entmax_ablBC.json'))
   B={'TWS_5f','TWM_5f','TWC_5f'}
   json.dump([x for x in r if x['dataset'] in B],  open('results/rerun_saint_entmax_ablB.json','w'))
   json.dump([x for x in r if x['dataset'] not in B], open('results/rerun_saint_entmax_ablC.json','w'))
   PY
   python scripts/merge_rerun_results.py --target results/ablation_b_noise.json --source results/rerun_saint_entmax_ablB.json --variants $V --tag $TAG
   python scripts/merge_rerun_results.py --target results/ablation_c_mk5.json   --source results/rerun_saint_entmax_ablC.json --variants $V --tag $TAG
   # Ablação A: os SEIS Transformers foram re-executados por transferência → substitui o arquivo inteiro
   cp results/ablation_a_transformers.json results/ablation_a_transformers_pre_${TAG}_backup.json
   cp results/rerun_saint_entmax_ablA.json results/ablation_a_transformers.json
   # Benchmark (Tabela 19): a fonte canônica é results/table19_results.json (era Downloads/table19_results.json)
   python scripts/merge_rerun_results.py --target results/table19_results.json --source results/rerun_saint_entmax_table19.json --variants SAINT_minibatch SAINT_fullbatch FTTransformer_entmax --tag $TAG
   python scripts/make_table19.py --input results/table19_results.json --output ../dissertacao-latex/tables/benchmark_transformers_memory.tex --n-values 1000,5000,10000,20000,50000
   # figuras do benchmark: scratch/regen_figures/plot_scaling.py (aponta para Downloads/table19_results.json — trocar para results/table19_results.json)
   ```
   Se `tier2_transformers_merged.json` ou os dumps `(1)/(2)` forem usados por algum gerador, aplicar o mesmo merge neles
   (o extrator `extract_tier2_fixed_params.py` lê `tier2_transformers*.json` — todos devem conter os registros novos, ou apagar os dumps antigos).
4. Regerar: `generate_analysis.py`, `generate_metrics_tables.py`, `generate_ablation_tables.py`, `generate_ablation_figs.py`,
   `generate_report_figs.py`, `generate_tier2_n5000_tables.py`/`analyze_tier2_n5000.py`, `compare_tabzilla.py`,
   `plot_decision_surfaces.py` (SAINT), `make_table19.py` (Tabela 19).
5. Copiar as tabelas/figuras para `dissertacao-latex/tables` e `Figuras`, e **reaplicar as edições manuais**:
   linhas dos Transformers em `tier2_sparsity.tex`; legenda longa de `benchmark_lssvm.tex`.
6. Revisar na dissertação os trechos com números de SAINT / FT-Entmax / Ablação A:
   Cap. Resultados (Tier 1 bullets, Friedman, Ablação A inteira, Ablação D §SAINT, Tier 2 bullets e esparsidade, benchmark inter-instâncias,
   validação TabZilla), Conclusão (itens 1, 5, 6), Resumo/Abstract se as conclusões mudarem, Apêndice (Nemenyi, métricas complementares).
7. Recompilar (`pdflatex ×2` + `bibtex`) e conferir o log.
