#!/bin/bash
# Benchmark de custo em 3 bases reais do Tier 2, m/N = 1% e 30% (2026-10-15).
cd "$(dirname "$0")"
PY=../../.venv/bin/python
for ds in adult credit bank; do
  for m in 0.01 0.30; do
    DATASET=$ds M_RATIO=$m $PY run_table18_benchmark.py --output real/${ds}_m${m}.json --repo-root ../.. \
      --variants NystromLSSVMColnorm,ADMMNystromLSSVM,FISTANystromLSSVM >> real/run.log 2>&1
  done
  DATASET=$ds $PY run_table18_benchmark.py --output real/${ds}_pcp.json --repo-root ../.. --variants PCPLSSVm >> real/run.log 2>&1
done
echo FIM >> real/run.log
