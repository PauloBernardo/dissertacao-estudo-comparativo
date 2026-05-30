#!/usr/bin/env python3
"""Tier 2 balanceado: testa H1 (imbalanceamento como causa do colapso).

Protocolo CORRIGIDO após code review (29/05/2026):
  - Usa datasets ORIGINAIS (CREDIT, BANK, SHOPPERS) — NÃO os CREDIT_BAL etc
  - Para cada seed:
      1. Carrega dataset original (imbalanceado)
      2. Subsample estratificado para N=7143
      3. Split 70/30 (preserva proporção original)
      4. Balanceia APENAS o treino via undersampling da majoritária
      5. Teste permanece como original (imbalanceado)
  - Resultado: comparação justa de "balanced training" vs "imbalanced training"
    sobre o MESMO conjunto de teste do Tier 2.

Versão anterior (bug): usava CREDIT_BAL/BANK_BAL/SHOPPERS_BAL que balanceavam
o dataset inteiro antes do split, contaminando o teste.

Output: results/tier2_balanced.json (separado de tier2_n5000.json)

Usage:
    # CPU group (todos os LSSVMs + SAINT)
    python scripts/run_tier2_balanced.py --group cpu_all

    # GPU group (4 FT baselines + FT-CUR)
    python scripts/run_tier2_balanced.py --group gpu_transformer
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# Datasets ORIGINAIS — o balanceamento é aplicado no treino pelo runner
DATASETS = ["CREDIT", "BANK", "SHOPPERS"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True,
                        choices=["cpu_all", "gpu_transformer", "all", "scalable"],
                        help="Grupo de modelos para rodar")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--skip-tuning", action="store_true",
                        help="Pula tuning Optuna (usa params do tier2_n5000)")
    args = parser.parse_args()

    base = Path("results")
    out_file = base / f"tier2_balanced_{args.group}.json"
    params_file = base / "tuning" / f"best_params_tier2_balanced_{args.group}.json"

    # Se --skip-tuning, usa params do tier2_n5000 imbalanceado como base
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
        "--balance-train",   # ← CHAVE: balanceia só treino, mantém teste original
    ]
    if args.skip_tuning:
        cmd.append("--skip-tuning")

    print("=" * 70)
    print("Tier 2 BALANCEADO — protocolo:")
    print("  - Datasets originais (imbalanceados):", DATASETS)
    print("  - Subsample N=7143 por seed (estratificado)")
    print("  - Split 70/30 (preserva proporção original)")
    print("  - Balanceamento APENAS no treino (undersampling da majoritária)")
    print("  - Teste preservado: mesma distribuição do Tier 2 imbalanceado")
    print("=" * 70)
    print()
    print("Comando:")
    print(" ", " ".join(cmd))
    print()
    # check=True: falha alto e claro se o subprocess sair com erro
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Subprocess falhou com código {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
