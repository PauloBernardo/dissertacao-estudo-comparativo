#!/usr/bin/env bash
# Continuação da fila l1rkhs (2026-10-14) após trocar o ρ do ADMM-Nyström para adaptativo
# (balanceamento de resíduos, Boyd et al. 2011 §3.4.1). Tier 1 do ADMM já em
# results/tier1_admm_adaptive.json; o FISTA-Nyström do Tier 1 e de B/C fica de tier1/ablation_bc_nystrom_l1rkhs.json.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=results/logs_proveniencia/admm_adaptive_queue.log; S30=$(seq 0 29)
step() { echo "=== $(date '+%F %T') $*" >> $L; }
step "Ablações B e C (ADMM)"
$PY scripts/run_tier1_gridcv.py --models ADMMNystromRandom --datasets TWS_5f TWM_5f TWC_5f MKE MKH MKM --seeds $S30 \
    --output results/ablation_bc_admm_adaptive.json --log-level WARNING >> $L 2>&1
step "Ablação A (ADMM)"
$PY scripts/run_ablation_a_retune.py --models ADMMNystromRandom --output results/ablation_a_admm_adaptive.json --log-level WARNING >> $L 2>&1
step "Ablação A (FISTA)"
$PY scripts/run_ablation_a_retune.py --models FISTANystromRandom --output results/ablation_a_fista_l1rkhs.json --log-level WARNING >> $L 2>&1
step "Tier 2"
$PY scripts/run_tier2_gridcv.py --models ADMMNystromRandom FISTANystromRandom --seeds $S30 --output results/tier2_nystrom_l1rkhs.json --log-level WARNING >> $L 2>&1
step "FIM"
