#!/usr/bin/env python3
"""Ablação A com re-tuning simétrico: GridSearchCV por semente em N=2000.

Decisão de 2026-10-06. O protocolo publicado da Ablação A misturava dois
regimes (8 modelos re-tunados com grades antigas e menores; os demais com
hiperparâmetros transferidos do Tier 1). Aqui TODOS os modelos passam pelo
mesmo protocolo do Tier 1 (hold-out 70/30, GridSearchCV 5-fold estratificado,
F1-macro), agora em TWS_2k/TWM_2k/TWC_2k, 20 sementes.

As grades são as que o Tier 1 publicado REALMENTE usou nos datasets TW, que em
dois casos diferem de src/tuning/grids.py:
  - ADMMNesterovLSSVM / ADMMElasticNet: lambda_ em {1.0, 0.1, 0.01} (45 combos);
    o GRIDS atual tem 5 valores (75).
  - NystromLSSVMRandom: grade de 96 -> rodar por scripts/run_nystrom_random_ablation.py.
Para os Transformers, passar --budget-epochs 200 --budget-patience 16.

Uso: python scripts/run_ablation_a_retune.py --models ... --output ... [--budget-...]
(os demais argumentos são os de run_tier1_gridcv.py; --datasets e --seeds têm
padrão próprio desta ablação)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.tuning.grids import GRIDS  # noqa: E402

TIER1_LAMBDA = [1.0, 0.1, 0.01]
for v in ("ADMMNesterovLSSVM", "ADMMElasticNet"):
    GRIDS[v]["grid"]["lambda_"] = TIER1_LAMBDA

import run_tier1_gridcv  # noqa: E402

if __name__ == "__main__":
    if "--datasets" not in sys.argv:
        sys.argv += ["--datasets", "TWS_2k", "TWM_2k", "TWC_2k"]
    if "--seeds" not in sys.argv:
        sys.argv += ["--seeds", *map(str, range(20))]
    if "NystromLSSVMRandom" in sys.argv:
        raise SystemExit("NystromLSSVMRandom: use run_nystrom_random_ablation.py (grade 96)")
    sys.exit(run_tier1_gridcv.main())
