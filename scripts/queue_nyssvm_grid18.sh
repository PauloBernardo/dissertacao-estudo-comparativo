#!/usr/bin/env bash
# Nyström-SVM na grade de 18 (σ{0,5;1,5;5} × γ, m/N=0,3) — a mesma do ADMM/FISTA-Nyström reformulados —
# no Tier 1, Ablações B/C e A (2026-10-14). Isola o efeito do ℓ1: os três Nyström só diferem pela penalidade.
# O Nyström-SVM canônico (grade de 96 nesses datasets) segue o publicado.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=results/logs_proveniencia/nyssvm_grid18.log
$PY scripts/run_tier1_gridcv.py --models NystromLSSVMRandom --seeds $(seq 0 29) --output results/nyssvm_grid18.json --log-level WARNING >> $L 2>&1
$PY scripts/run_tier1_gridcv.py --models NystromLSSVMRandom --datasets TWS_5f TWM_5f TWC_5f MKE MKH MKM --seeds $(seq 0 29) --output results/nyssvm_grid18.json --log-level WARNING >> $L 2>&1
$PY scripts/run_tier1_gridcv.py --models NystromLSSVMRandom --datasets TWS_2k TWM_2k TWC_2k --seeds $(seq 0 19) --output results/nyssvm_grid18.json --log-level WARNING >> $L 2>&1
echo "=== FIM $(date '+%F %T')" >> $L
