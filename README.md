# Sparse LSSVM vs Sparse Tabular Transformers

Estudo experimental sobre classificação tabular binária comparando variantes esparsas de LSSVM, Transformers tabulares e baselines clássicos.

## Estado atual do repositório

O fluxo ativo e verificável no código atual é:

1. baixar ou validar os datasets com `scripts/download_data.py`
2. executar o protocolo hold-out + `GridSearchCV` com `scripts/run_tier1_gridcv.py`
3. executar ou validar o Tier 2 com `scripts/run_tier2_gridcv.py` e os resultados Transformer consolidados
4. analisar os artefatos em `results/`

Referências no histórico a `run_tuning_tier1.py`, `run_experiments_tier1.py` e reruns pontuais por modelo permanecem em `OLD/` como material arquivado, não como entrypoints atuais.

## Estrutura

```text
config/                  YAMLs auxiliares
notebooks/               notebooks de execução e análise
results/                 artefatos já gerados
scripts/
  download_data.py       baixa e valida datasets
  run_tier1_gridcv.py    protocolo atual do Tier 1
  run_tier2_gridcv.py    protocolo atual do Tier 2 CPU
src/
  data/                  loaders, sintéticos e pré-processamento
  experiments/           runner e reprodutibilidade
  metrics/               métricas de performance, esparsidade e eficiência
  models/                implementações dos modelos
  tuning/                grids e tuning helpers
tests/                   suíte pytest
OLD/                     scripts e relatórios históricos
```

Hoje a intenção é que `scripts/` fique enxuto e operacional. Scripts de correção histórica, rerun e ablações específicas foram mantidos em `OLD/scripts/`.

## Ambiente

Requisitos práticos:

- Python 3.11+
- GPU opcional para variantes baseadas em Transformer

Setup recomendado:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,transformers]"
```

`xgboost` é dependência formal do projeto e é usado por [src/models/xgboost_wrapper.py](/home/paulo/Documentos/dissertacao-estudo-comparativo/sparse-lssvm-transformers-study/src/models/xgboost_wrapper.py).

## Datasets

O script abaixo baixa e valida os datasets do Tier 1. Os dados ficam em `data/raw/`.

```bash
python scripts/download_data.py --tier 1
```

Tier 1 usa os datasets reais `BCW`, `PID`, `HAB`, `VCP`, `GCR`, `AUS` e os sintéticos `TWS`, `TWM`, `TWC`.

## Testes

```bash
pytest tests/ -q
```

Alguns testes dependem de rede porque certos loaders baixam datasets externos na primeira execução.

## Tier 1

O protocolo atual está implementado em [scripts/run_tier1_gridcv.py](/home/paulo/Documentos/dissertacao-estudo-comparativo/sparse-lssvm-transformers-study/scripts/run_tier1_gridcv.py):

- split estratificado 70/30 por seed
- `GridSearchCV` com `StratifiedKFold(5, shuffle=True, random_state=seed)` no treino
- `StandardScaler` dentro do `Pipeline`
- `refit=True` no melhor conjunto de hiperparâmetros
- avaliação única no hold-out de teste
- execução resumível por JSON

### Smoke test

```bash
python scripts/run_tier1_gridcv.py \
  --models StandardLSSVM XGBoost \
  --datasets BCW PID \
  --seeds 0 1 \
  --output results/tier1_smoketest.json \
  --log-level INFO
```

### Execução completa

```bash
python -u scripts/run_tier1_gridcv.py \
  --output results/tier1_gridcv.json \
  --log-level INFO
```

Por padrão, o script roda todos os variants definidos em [src/tuning/grids.py](/home/paulo/Documentos/dissertacao-estudo-comparativo/sparse-lssvm-transformers-study/src/tuning/grids.py) sobre os 9 datasets do Tier 1 e 30 seeds.

### Rodar só variantes CPU

```bash
python scripts/run_tier1_gridcv.py \
  --models StandardLSSVM PCPLSSVm FSALSSVm IPLSSVm PruningLSSVM OppositeMapsLSSVM \
           ADMMNesterovLSSVM ADMMElasticNet FISTANesterov DualFISTA \
           NystromLSSVMColnorm XGBoost \
  --output results/tier1_cpu.json \
  --log-level INFO
```

### Rodar só variantes que pedem GPU

```bash
python scripts/run_tier1_gridcv.py \
  --models FTTransformer_softmax FTTransformer_topk FTTransformer_entmax FTTransformer_sparsemax \
           SAINTColnorm FTTransformerCURColnorm \
  --output results/tier1_gpu.json \
  --log-level INFO
```

## Saídas

Arquivos gerados ou reaproveitados com frequência:

- `results/tier1_gridcv.json`: saída principal do protocolo atual
- `results/tier2_gridcv.json`: Tier 2 CPU, 9 modelos × 6 datasets × 30 sementes
- `results/tier2_transformers.json`: Tier 2 Transformer, 6 modelos × 6 datasets × 30 sementes
- `results/tier2_combined.json`: concatenação reprodutível dos dois artefatos oficiais do Tier 2
- `results/tuning/*.json`: snapshots de tuning e reruns específicos
- `results/tables/`: tabelas LaTeX
- `results/plots/`: figuras em PNG e PDF
- `results/relatorio.tex` e `results/relatorio.pdf`: relatório consolidado

Os arquivos `results/tier1_results.json` e `results/tier1_custom_models.json` existem como artefatos históricos relevantes, mas não são a saída padrão do entrypoint atual.

## Tier 2

O Tier 2 consolidado usado na dissertação combina:

- `results/tier2_gridcv.json`, gerado pelo fluxo CPU com LSSVMs e XGBoost
- `results/tier2_transformers.json`, resultado Transformer completo após a correção do merge para preservar as 6 variantes e o rerun do FT-CUR com pseudo-inversa de Newton-Schulz

O arquivo `results/tier2_transformers_merged.json` é histórico e incompleto (540 registros); não deve ser usado para tabelas finais.

## Notas metodológicas

- Modelos LSSVM usam rótulos assinados internamente quando necessário.
- `SAINTColnorm` e `FTTransformerCURColnorm` estão configurados no grid atual com `early_stop_metric="val_loss"`.
- O script é resiliente a interrupção: grava cada registro de forma atômica e pode ser retomado.

## Histórico

O diretório `OLD/` contém scripts, relatórios e artefatos de fases anteriores do projeto. Use-o como referência histórica, não como documentação operacional primária.
