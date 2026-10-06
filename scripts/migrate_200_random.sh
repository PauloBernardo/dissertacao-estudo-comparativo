#!/usr/bin/env bash
# Migração 2026-10-06: protocolo 200/16 para os Transformers e seletor `random` no lugar de `colnorm`
# (FT-CUR, LSSVM-Nyström, ADMM-Nyström, FISTA-Nyström). Ver docs/PLANO_FECHAMENTO.md, "PLANO DE MIGRAÇÃO".
# Cada merge num mesmo alvo usa tag distinta: o backup *_pre_200_backup.json é sempre o original.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
M="$PY scripts/merge_rerun_results.py"
TF="FTTransformer_softmax FTTransformer_topk FTTransformer_entmax FTTransformer_sparsemax SAINTColnorm FTTransformerCURRandom"

# Tier 1
$M --target results/tier1_gridcv.json --source results/tier1_principal_200.json --variants $TF --remove FTTransformerCURColnorm --tag 200
$M --target results/tier1_gridcv.json --source results/tier1_nystrom_random.json --variants NystromLSSVMRandom --remove NystromLSSVMColnorm --tag 200b
$M --target results/tier1_gridcv.json --source results/tier1_nystrom_lssvm_random.json --variants ADMMNystromRandom FISTANystromRandom --remove ADMMNystromLSSVM FISTANystrom --tag 200c
# Tier 2
$M --target results/tier2_transformers.json --source results/tier2_principal_200.json --variants $TF --remove FTTransformerCURColnorm --tag 200
$M --target results/tier2_gridcv.json --source results/tier2_nystrom_random.json --variants NystromLSSVMRandom --remove NystromLSSVMColnorm --tag 200
$M --target results/tier2_gridcv.json --source results/tier2_nystrom_lssvm_random.json --variants ADMMNystromRandom FISTANystromRandom --remove ADMMNystromLSSVM FISTANystrom --tag 200b
# N=5000 (Ablação D)
$M --target results/tier2_fixedparams_n5000_transformers.json --source results/ablD_200.json --variants $TF --remove FTTransformerCURColnorm --tag 200
$M --target results/tier2_fixedparams_n5000_lssvm.json --source results/tier2_fixedparams_n5000_nystrom_random.json --variants NystromLSSVMRandom --remove NystromLSSVMColnorm --tag 200
$M --target results/tier2_fixedparams_n5000_lssvm.json --source results/tier2_fixedparams_n5000_nystrom_lssvm_random.json --variants ADMMNystromRandom FISTANystromRandom --remove ADMMNystromLSSVM FISTANystrom --tag 200b
# Ablação A: substituição inteira
cp results/ablation_a_transformers.json results/ablation_a_transformers_pre_200_backup.json
cp results/ablation_a_200.json results/ablation_a_transformers.json
# Ablações B e C: separar por dataset
$PY - <<'PYX'
import json
r=json.load(open('results/ablation_bc_200.json')); B={'TWS_5f','TWM_5f','TWC_5f'}
json.dump([x for x in r if x['dataset'] in B], open('results/ablation_b_200.json','w'))
json.dump([x for x in r if x['dataset'] not in B], open('results/ablation_c_200.json','w'))
PYX
$M --target results/ablation_b_noise.json --source results/ablation_b_200.json --variants $TF --remove FTTransformerCURColnorm --tag 200
$M --target results/ablation_c_mk5.json --source results/ablation_c_200.json --variants $TF --remove FTTransformerCURColnorm --tag 200
# Tabela 19
$M --target results/table19_results.json --source results/table19_200.json --variants SAINT_minibatch SAINT_fullbatch FTCUR_minibatch FTCUR_mfixed_full --tag 200
# Ablações A (LSSVMs), B e C: Nyström-LSSVM, ADMM- e FISTA-Nyström reexecutados com `random`
# (2026-10-06; logs em results/logs_proveniencia/abl{A,BC}_nystrom_random.log)
NR="NystromLSSVMRandom ADMMNystromRandom FISTANystromRandom"; NC="NystromLSSVMColnorm ADMMNystromLSSVM FISTANystrom"
$PY - <<'PYX'
import json
# ADMM/FISTA por run_tier1_gridcv; Nyström-SVM por run_nystrom_random_ablation (grade 96 e rótulos ±1)
r=json.load(open('results/ablation_bc_nystrom_random.json'))+json.load(open('results/ablation_bc_nystromsvm_random.json')); B={'TWS_5f','TWM_5f','TWC_5f'}
json.dump([x for x in r if x['dataset'] in B], open('results/ablation_b_nystrom_random.json','w'))
json.dump([x for x in r if x['dataset'] not in B], open('results/ablation_c_nystrom_random.json','w'))
PYX
$M --target results/ablation_b_noise.json --source results/ablation_b_nystrom_random.json --variants $NR --remove $NC --tag 200b
$M --target results/ablation_c_mk5.json --source results/ablation_c_nystrom_random.json --variants $NR --remove $NC --tag 200b
# Ablação A: re-tuning simétrico em N=2000 (decisão de 2026-10-06; ver PLANO_FECHAMENTO.md). Substituição
# INTEIRA dos dois arquivos, como em setembro. As reexecuções por transferência (ablation_a_nystrom_random,
# ablation_a_transfer_rerun) foram descartadas e não entram.
$PY - <<'PYX'
import json, shutil
for alvo, fontes in (('results/ablation_a_scaling.json', ['results/ablation_a_retune_cpu.json', 'results/ablation_a_retune_nystromsvm.json']),
                     ('results/ablation_a_transformers.json', ['results/ablation_a_retune_tf.json'])):
    recs = [r for f in fontes for r in json.load(open(f)) if r.get('status') == 'ok']
    shutil.copy(alvo, alvo.replace('.json', '_pre_retune_backup.json'))
    json.dump(recs, open(alvo, 'w'), indent=2); print(alvo, len(recs))
PYX
