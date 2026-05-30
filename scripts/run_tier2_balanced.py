#!/usr/bin/env python3
"""Tier 2 balanceado: testa a hipótese de imbalanceamento como causa do colapso.

Datasets:
  - CREDIT_BAL    (13272 amostras, 50/50)  — originalmente 22% positivos
  - BANK_BAL      (10578 amostras, 50/50)  — originalmente 11%
  - SHOPPERS_BAL  ( 3816 amostras, 50/50)  — originalmente 15%

Random undersampling determinístico (seed=42) da classe majoritária.

Protocolo: mesmo do Tier 2 N=5000:
  - Subsample estratificado para N=7143 por seed (90/30 split → ~5000 train)
  - Tuning Optuna 20 trials × 3-fold CV
  - 30 seeds para experimentos

Para evitar conflito com Tier 2 N=5000:
  - Output: results/tier2_balanced.json
  - Params: results/tuning/best_params_tier2_balanced.json

Usage:
    # Lança ambos grupos (CPU + GPU) em paralelo, como fizemos antes:

    # Terminal 1 — Script CPU (todos os 13 modelos LSSVMs + SAINT)
    python scripts/run_tier2_balanced.py --group cpu_all

    # Terminal 2 — Script GPU (4 FT baselines + FT-CUR)
    python scripts/run_tier2_balanced.py --group gpu_transformer
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DATASETS = ["CREDIT_BAL", "BANK_BAL", "SHOPPERS_BAL"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True,
                        choices=["cpu_all", "gpu_transformer", "all", "scalable"],
                        help="Grupo de modelos para rodar")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--skip-tuning", action="store_true",
                        help="Pula tuning Optuna (usa params do tier2_n5000 imbalanceado)")
    args = parser.parse_args()

    base = Path("results")
    out_file = base / f"tier2_balanced_{args.group}.json"
    params_file = base / "tuning" / f"best_params_tier2_balanced_{args.group}.json"

    # Se --skip-tuning, usa os params do tier2 imbalanceado como base
    # (não é rigoroso mas é rápido para validar a hipótese inicialmente)
    if args.skip_tuning:
        src_params = base / "tuning" / f"best_params_tier2_n5000_{args.group}.json"
        if src_params.exists() and not params_file.exists():
            print(f"[INFO] Copiando params do tier2_n5000 → tier2_balanced (reuso)")
            params_file.write_bytes(src_params.read_bytes())

    cmd = [
        sys.executable, "scripts/run_tier2_n5000.py",
        "--seeds", str(args.seeds),
        "--trials", str(args.trials),
        "--folds", str(args.folds),
        "--models-group", args.group,
        "--datasets", *DATASETS,
        "--output-file", str(out_file),
        "--params-file", str(params_file),
    ]
    if args.skip_tuning:
        cmd.append("--skip-tuning")

    print("Lançando:")
    print(" ", " ".join(cmd))
    print()
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
