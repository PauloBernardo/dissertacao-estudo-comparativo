#!/usr/bin/env bash
# Reexecução do LSSVM-PCP fiel a Zhou (2016), Alg. 1 + Eq. (11) (2026-10-15): sem matriz N×N,
# diagonal exata, argmax simples, parada pelo traço do resíduo, α_B e b conjuntos.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=results/logs_proveniencia/pcp_fiel_queue.log; S30=$(seq 0 29)
step() { echo "=== $(date '+%F %T') $*" >> $L; }
step "Tier 1"
$PY scripts/run_tier1_gridcv.py --models PCPLSSVm --seeds $S30 --output results/tier1_pcp_fiel.json --log-level WARNING >> $L 2>&1
step "Ablações B e C"
$PY scripts/run_tier1_gridcv.py --models PCPLSSVm --datasets TWS_5f TWM_5f TWC_5f MKE MKH MKM --seeds $S30 \
    --output results/ablation_bc_pcp_fiel.json --log-level WARNING >> $L 2>&1
step "Ablação A"
$PY scripts/run_ablation_a_retune.py --models PCPLSSVm --output results/ablation_a_pcp_fiel.json --log-level WARNING >> $L 2>&1
step "Tier 2"
$PY scripts/run_tier2_gridcv.py --models PCPLSSVm --seeds $S30 --output results/tier2_pcp_fiel.json --log-level WARNING >> $L 2>&1
step "Moda do PCP no config (só a entrada do PCP)"
$PY scripts/update_fixed_params_variant.py --source results/tier2_pcp_fiel.json --variant PCPLSSVm --tag pre_pcp_fiel >> $L 2>&1
step "Ablação D"
$PY scripts/run_tier2_fixedparams.py --models PCPLSSVm --n-train 5000 --seeds $S30 \
    --output results/tier2_fixedparams_n5000_pcp_fiel.json --log-level WARNING >> $L 2>&1
step "FIM"
