#!/usr/bin/env bash
# Mescla o refit dos Transformers sem paciência (scripts/refit_tf_nopatience.py, 2026-10-14) nos JSONs
# canônicos. Snapshot do protocolo anterior (200/16) em results/protocolo_pat16/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
TF="FTTransformer_softmax FTTransformer_topk FTTransformer_entmax FTTransformer_sparsemax SAINTColnorm FTTransformerCURRandom"
$PY - <<'PYX'
import json
for k in ('tier1', 'ablA', 'ablB', 'ablC'):
    r = [x for p in ('p0', 'p1') for x in json.load(open(f'results/refit_{k}_{p}.json')) if x['status'] == 'ok']
    json.dump(r, open(f'results/refit_{k}.json', 'w')); print(k, len(r))
PYX
for spec in "tier1_gridcv:tier1" "ablation_a_transformers:ablA" "ablation_b_noise:ablB" "ablation_c_mk5:ablC" \
            "tier2_transformers:tier2" "tier2_fixedparams_n5000_transformers:ablD"; do
  $PY scripts/merge_rerun_results.py --target results/${spec%%:*}.json --source results/refit_${spec#*:}.json \
      --variants $TF --tag prerefit
done
