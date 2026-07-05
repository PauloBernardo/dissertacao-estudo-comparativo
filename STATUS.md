# STATUS

Atualizado em 04/07/2026.

## Resumo

O repositório contém código ativo para execução do Tier 1, execução CPU do Tier 2, suítes de teste, artefatos consolidados de resultados e material histórico em `OLD/`.

## Estado verificado no código

- O entrypoint atual de Tier 1 é `scripts/run_tier1_gridcv.py`.
- O entrypoint CPU atual de Tier 2 é `scripts/run_tier2_gridcv.py`.
- O download e a validação dos datasets ficam em `scripts/download_data.py`.
- O conjunto de variantes e grids ativos está em `src/tuning/grids.py`.
- Os scripts `run_tuning_tier1.py`, `run_experiments_tier1.py` e reruns especializados por modelo foram arquivados em `OLD/scripts/`.
- Há artefatos históricos em `results/` que não correspondem necessariamente à saída padrão do fluxo atual; para o Tier 2 final, use `tier2_gridcv.json`, `tier2_transformers.json` e `tier2_combined.json`.

## Dependências e ambiente

- `pyproject.toml` pede Python `>=3.11`.
- Para os modelos Transformer, o setup prático é `pip install -e ".[dev,transformers]"`.
- O baseline `XGBoost` é dependência formal em `pyproject.toml` e `requirements.txt`.

## Tier 1

Fluxo atual confirmado:

1. `python scripts/download_data.py --tier 1`
2. `pytest tests/ -q`
3. `python scripts/run_tier1_gridcv.py --output results/tier1_gridcv.json --log-level INFO`

Características do protocolo:

- hold-out estratificado 70/30 por seed
- `GridSearchCV` com 5 folds no treino
- `StandardScaler` dentro de `Pipeline`
- `refit=True`
- execução resumível por JSON

## Tier 2

Estado consolidado usado na dissertação:

1. `results/tier2_gridcv.json`: 1620 execuções CPU completas (9 modelos × 6 datasets × 30 sementes).
2. `results/tier2_transformers.json`: 1080 execuções Transformer completas (6 modelos × 6 datasets × 30 sementes).
3. `results/tier2_combined.json`: concatenação dos dois artefatos oficiais, totalizando 2700 execuções e 15 modelos.

O arquivo `results/tier2_transformers_merged.json` é histórico e incompleto (540 execuções); não deve ser usado para tabelas finais.

## Limpeza documental concluída

- `README.md` atualizado para refletir o fluxo real do Tier 1 e do Tier 2 consolidado
- `scripts/` reduzido aos entrypoints operacionais atuais
- referências a scripts arquivados removidas da documentação principal
- `STATUS.md` reduzido a um snapshot estável, sem PIDs, ETAs e comandos transitórios

## Pendências recomendadas

1. Decidir se os reruns Transformer de Tier 2 serão transformados em um entrypoint único ou continuarão como artefatos consolidados.
2. Separar de forma explícita, na documentação, resultados históricos de resultados reproduzíveis pelo fluxo atual.
