#!/bin/bash
# Piloto do Tier 3: um fit por (modelo, m) em HIGGS500K, semente 0, CPU sozinha.
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=""
OUT=results/pilot_tier3.json
LOG=results/logs_proveniencia/pilot_tier3.log
for v in PCPLSSVm:2000 ADMMNystromRandom:2000 FISTANystromRandom:2000 \
         NystromLSSVMRandom:5000 PCPLSSVm:5000 ADMMNystromRandom:5000 FISTANystromRandom:5000; do
  echo "== ${v} $(date +%T)" >> $LOG
  ( ulimit -v 14000000; .venv/bin/python scripts/pilot_tier3.py --variant ${v%:*} --m ${v#*:} \
      --seed 0 --output $OUT ) >> $LOG 2>&1 || echo "FALHOU ${v} (rc=$?)" >> $LOG
done
echo FIM >> $LOG
