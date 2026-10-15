#!/usr/bin/env bash
# Mescla a reexecução dos 3 FISTA com o restart de O'Donoghue–Candès corrigido (2026-10-13;
# fila scripts/queue_fista_rerun.sh, log results/logs_proveniencia/fista_rerun.log) nos JSONs canônicos.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
M="$PY scripts/merge_rerun_results.py"
F3="DualFISTA FISTANesterov FISTANystromRandom"
$PY - <<'PYX'
import json
t2 = json.load(open('results/rerun_fista_tier2.json')) + json.load(open('results/rerun_fista_tier2_s20-29.json'))
json.dump(t2, open('results/rerun_fista_tier2_all.json', 'w'))
r = json.load(open('results/rerun_fista_ablbc.json')); B = {'TWS_5f', 'TWM_5f', 'TWC_5f'}
json.dump([x for x in r if x['dataset'] in B], open('results/rerun_fista_ablb.json', 'w'))
json.dump([x for x in r if x['dataset'] not in B], open('results/rerun_fista_ablc.json', 'w'))
n = json.load(open('results/rerun_fista_n5000.json'))
for v in ('DualFISTA', 'FISTANystromRandom'):
    json.dump([x for x in n if x['variant'] == v], open(f'results/rerun_fista_n5000_{v}.json', 'w'))
PYX
$M --target results/tier1_gridcv.json --source results/rerun_fista_tier1.json --variants $F3 --tag prefista
$M --target results/tier2_gridcv.json --source results/rerun_fista_tier2_all.json --variants DualFISTA FISTANystromRandom --tag prefista
$M --target results/ablation_b_noise.json --source results/rerun_fista_ablb.json --variants $F3 --tag prefista
$M --target results/ablation_c_mk5.json --source results/rerun_fista_ablc.json --variants $F3 --tag prefista
$M --target results/tier2_fixedparams_n5000_lssvm.json --source results/rerun_fista_n5000_DualFISTA.json --variants DualFISTA --tag prefista
$M --target results/tier2_fixedparams_n5000_lssvm.json --source results/rerun_fista_n5000_FISTANystromRandom.json --variants FISTANystromRandom --tag prefista_b
