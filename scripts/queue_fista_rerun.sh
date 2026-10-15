#!/usr/bin/env bash
# Reexecução dos três FISTA com o restart corrigido (2026-10-13), com o protocolo publicado de cada fase.
# Retomável. Log em results/logs_proveniencia/fista_rerun.log.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=results/logs_proveniencia/fista_rerun.log
step(){ echo "=== $(date '+%F %T') $*" >> $L; }
step "Ablação D (N=5000, config fixa reconstruída dos registros publicados)"
$PY scripts/run_tier2_fixedparams.py --models DualFISTA FISTANystromRandom --n-train 5000 \
    --config config/tier2_fixed_params_fista_rerun.json --output results/rerun_fista_n5000.json --log-level WARNING >> $L 2>&1
step "Ablações B e C (grades de junho)"
$PY scripts/run_fista_rerun_bc.py --models DualFISTA FISTANystromRandom FISTANesterov \
    --output results/rerun_fista_ablbc.json --log-level WARNING >> $L 2>&1
step "Tier 1"
$PY scripts/run_tier1_gridcv.py --models DualFISTA FISTANystromRandom FISTANesterov \
    --output results/rerun_fista_tier1.json --log-level WARNING >> $L 2>&1
step "Tier 2"
$PY scripts/run_tier2_gridcv.py --models DualFISTA FISTANystromRandom --n-train 2000 \
    --output results/rerun_fista_tier2.json --log-level WARNING >> $L 2>&1
step "FIM"
