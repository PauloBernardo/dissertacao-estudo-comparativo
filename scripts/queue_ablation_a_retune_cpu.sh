#!/usr/bin/env bash
# Fila de CPU do re-tuning simétrico da Ablação A (2026-10-06). Retomável: rodar de novo pula o que já existe.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=results/logs_proveniencia
$PY scripts/run_nystrom_random_ablation.py --datasets TWS_2k TWM_2k TWC_2k --seeds $(seq 0 19) \
    --output results/ablation_a_retune_nystromsvm.json --log-level WARNING >> $L/ablA_retune_cpu.log 2>&1
for m in XGBoost PCPLSSVm StandardLSSVM FISTANystromRandom ADMMNystromRandom OppositeMapsOriginalLSSVM \
         IPLSSVmOriginal PruningLSSVM DualFISTA FISTANesterov ADMMNesterovLSSVM ADMMElasticNet FSALSSVmOriginal; do
  echo "=== $(date '+%F %T') $m" >> $L/ablA_retune_cpu.log
  $PY scripts/run_ablation_a_retune.py --models $m --output results/ablation_a_retune_cpu.json \
      --log-level WARNING >> $L/ablA_retune_cpu.log 2>&1 || echo "FALHOU $m" >> $L/ablA_retune_cpu.log
done
echo "=== $(date '+%F %T') FIM" >> $L/ablA_retune_cpu.log
