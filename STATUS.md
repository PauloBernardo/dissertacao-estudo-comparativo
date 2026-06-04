# STATUS

Atualizado em 03/06/2026.

## Resumo

O repositório contém código ativo para execução do Tier 1, suítes de teste, artefatos de resultados e material histórico em `OLD/`. A principal inconsistência encontrada era documental: o `README.md` apontava scripts arquivados como se fossem o fluxo atual.

## Estado verificado no código

- O entrypoint atual de Tier 1 é `scripts/run_tier1_gridcv.py`.
- O download e a validação dos datasets ficam em `scripts/download_data.py`.
- O conjunto de variantes e grids ativos está em `src/tuning/grids.py`.
- Os scripts `run_tuning_tier1.py`, `run_experiments_tier1.py` e reruns especializados por modelo foram arquivados em `OLD/scripts/`.
- Há artefatos históricos em `results/` que não correspondem necessariamente à saída padrão do fluxo atual.

## Dependências e ambiente

- `pyproject.toml` pede Python `>=3.11`.
- Para os modelos Transformer, o setup prático é `pip install -e ".[dev,transformers]"`.
- O baseline `XGBoost` é usado pelo código ativo, mas `xgboost` não está declarado hoje em `pyproject.toml` nem em `requirements.txt`.

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

O repositório ainda contém notebooks, resultados e notas relacionadas ao Tier 2, mas os scripts operacionais principais citados em versões anteriores da documentação não estão presentes no diretório `scripts/` atual. Na prática:

- use os notebooks em `notebooks/` e os artefatos em `results/` como referência
- trate instruções antigas sobre Tier 2 como histórico até nova consolidação

## Limpeza documental concluída

- `README.md` atualizado para refletir o fluxo real do Tier 1
- `scripts/` reduzido aos entrypoints operacionais atuais
- referências a scripts arquivados removidas da documentação principal
- `STATUS.md` reduzido a um snapshot estável, sem PIDs, ETAs e comandos transitórios

## Pendências recomendadas

1. Declarar `xgboost` formalmente em `pyproject.toml` e `requirements.txt`.
2. Decidir se o Tier 2 terá novos entrypoints locais em `scripts/` ou se os notebooks serão a interface oficial.
3. Separar de forma explícita, na documentação, resultados históricos de resultados reproduzíveis pelo fluxo atual.
