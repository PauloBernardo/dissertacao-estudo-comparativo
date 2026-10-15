#!/usr/bin/env bash
# Fila de CPU da reformulação do ADMM-/FISTA-Nyström com o regularizador do LSSVM (2026-10-14):
# (1/2τ)‖Ĉθ − ŷ‖² + ½θᵀWθ + λ‖θ‖₁, grade σ,τ do Nyström-SVM + lambda_rel. Retomável.
# A Ablação D (parâmetros fixos da moda do Tier 2) roda depois, quando o Tier 2 terminar.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=results/logs_proveniencia/l1rkhs_queue.log
V="ADMMNystromRandom FISTANystromRandom"; S30=$(seq 0 29)
step() { echo "=== $(date '+%F %T') $*" >> $L; }
step "Tier 1"
$PY scripts/run_tier1_gridcv.py --models $V --seeds $S30 --output results/tier1_nystrom_l1rkhs.json --log-level WARNING >> $L 2>&1
step "Ablações B e C"
$PY scripts/run_tier1_gridcv.py --models $V --datasets TWS_5f TWM_5f TWC_5f MKE MKH MKM --seeds $S30 \
    --output results/ablation_bc_nystrom_l1rkhs.json --log-level WARNING >> $L 2>&1
step "Ablação A"
for m in $V; do
  $PY scripts/run_ablation_a_retune.py --models $m --output results/ablation_a_nystrom_l1rkhs.json --log-level WARNING >> $L 2>&1
done
step "Tier 2"
$PY scripts/run_tier2_gridcv.py --models $V --seeds $S30 --output results/tier2_nystrom_l1rkhs.json --log-level WARNING >> $L 2>&1
step "FIM"
