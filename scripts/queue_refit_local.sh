#!/usr/bin/env bash
# Refit sem paciência (scripts/refit_tf_nopatience.py), parte local: Tier 1 e Ablações A, B, C.
# Dois processos por bloco (sementes pares/ímpares) para ocupar a GPU; shards _p0/_p1.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=results/logs_proveniencia/refit_local.log
EV=$(seq 0 2 29); OD=$(seq 1 2 29)
for spec in "tier1:results/tier1_gridcv.json" "ablA:results/ablation_a_transformers.json" \
            "ablB:results/ablation_b_noise.json" "ablC:results/ablation_c_mk5.json"; do
  k=${spec%%:*}; src=${spec#*:}
  echo "=== $(date '+%F %T') $k" >> $L
  $PY scripts/refit_tf_nopatience.py --source $src --seeds $EV --output results/refit_${k}_p0.json >> $L 2>&1 &
  $PY scripts/refit_tf_nopatience.py --source $src --seeds $OD --output results/refit_${k}_p1.json >> $L 2>&1
  wait
done
echo "=== $(date '+%F %T') FIM" >> $L
