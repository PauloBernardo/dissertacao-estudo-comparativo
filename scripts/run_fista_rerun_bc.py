#!/usr/bin/env python3
"""Reexecução dos FISTA nas Ablações B e C com as grades PUBLICADAS (2026-10-13).

As Ablações B/C rodaram em junho, com as grades do commit 900f551 de src/tuning/grids.py, que
diferem das atuais para dois modelos:
  - FISTANesterov: sigma {0,05; 0,15; 0,5; 1,5; 5} x tau {0,01; 0,1; 1; 10} x lambda {0,001; 0,005; 0,01; 0,1} = 80
  - DualFISTA:     sigma {0,05; 0,5; 5; 50} x tau {0,01; 0,1; 1; 10} x lambda {0,001; 0,01; 0,1} = 48
(conferido contra grid_size e best_params de results/ablation_b_noise.json e ablation_c_mk5.json).
O FISTA-Nyström usou a grade atual (45). Motivo da reexecução: o restart dos FISTA estava errado.
Uso: python scripts/run_fista_rerun_bc.py --models ... --output ... (demais args de run_tier1_gridcv.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.tuning.grids import GRIDS  # noqa: E402

GRIDS["FISTANesterov"]["grid"] = {"sigma": [0.05, 0.15, 0.5, 1.5, 5.0], "tau": [0.01, 0.1, 1.0, 10.0],
                                  "lambda_": [0.001, 0.005, 0.01, 0.1]}
GRIDS["DualFISTA"]["grid"] = {"sigma": [0.05, 0.5, 5.0, 50.0], "tau": [0.01, 0.1, 1.0, 10.0],
                              "lambda_": [0.001, 0.01, 0.1]}
import run_tier1_gridcv  # noqa: E402

if __name__ == "__main__":
    if "--datasets" not in sys.argv:
        sys.argv += ["--datasets", "TWS_5f", "TWM_5f", "TWC_5f", "MKE", "MKM", "MKH"]
    sys.exit(run_tier1_gridcv.main())
