#!/usr/bin/env python3
"""Script C: roda experimentos dos modelos já tunados em PARALELO ao Script A.

Estratégia para acelerar o Tier 2 N=5000:
  - Script A (em execução) ainda tuna ADMM-Nesterov e ADMM-ElasticNet.
  - CPU está ocioso (~50% livre).
  - Este Script C usa os params já tunados para rodar experimentos
    dos 10 modelos que já têm tuning completo (6/6 datasets).

Modelos rodados:
  Std, PCP, FSA, IP, Pruning, OppM, FISTA-Nesterov,
  DualFISTA, Nyström-SVM, SAINT.
  (XGBoost já tem 180 runs no checkpoint principal.)

Segurança:
  - Lê params de um SNAPSHOT (cópia) para evitar race com Script A
  - Escreve em arquivo SEPARADO (tier2_n5000_parallel.json)
  - Merge feito manualmente no final

Usage:
    # 1. Tirar snapshot do params atual:
    cp results/tuning/best_params_tier2_n5000_cpu.json \\
       results/tuning/best_params_tier2_n5000_snapshot.json

    # 2. Lançar Script C:
    python scripts/run_parallel_tuned.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/tier2_parallel.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Configuração ──────────────────────────────────────────────────────────────

N_TOTAL_CAP = 7143   # mesmo cap do Script A
DATASETS = ["ADULT", "CREDIT", "BANK", "TELCO", "SHOPPERS", "HIGGS50K"]

PARAMS_SNAPSHOT = Path("results/tuning/best_params_tier2_n5000_snapshot.json")
OUTPUT_FILE     = Path("results/tier2_n5000_parallel.json")

# Modelos a rodar (já totalmente tunados). XGBoost é excluído (já em outro arquivo).
# Ordem: começa pelos mais rápidos para ter resultados parciais cedo
MODELS_TO_RUN = [
    ("NystromLSSVMColnorm",       "NystromLSSVMColnorm"),   # rápido
    ("StandardLSSVM",             "StandardLSSVM"),
    ("PCPLSSVm",                  "PCPLSSVm"),
    ("IPLSSVm",                   "IPLSSVm"),
    ("PruningLSSVM",              "PruningLSSVM"),
    ("OppositeMapsLSSVM",         "OppositeMapsLSSVM"),
    ("FSALSSVm",                  "FSALSSVm"),
    ("FISTANesterovLSSVM",        "FISTANesterov"),
    ("DualFISTALSSVM",            "DualFISTA"),
    ("SAINTColnorm",              "SAINTColnorm"),          # mais lento (CPU)
]


def _save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def _existing_keys(results):
    keys = set()
    for r in results:
        mv = r.get("model_variant") or r.get("model")
        keys.add(f"{mv}__{r.get('dataset')}__{r.get('seed')}")
    return keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--params-snapshot", type=Path, default=PARAMS_SNAPSHOT,
                        help="Snapshot do best_params (cópia para evitar race)")
    parser.add_argument("--output-file", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()

    from src.experiments.runner import run_single_experiment

    seeds = list(range(args.seeds))

    log.info("=" * 70)
    log.info("Script C — Experimentos paralelos dos modelos já tunados")
    log.info("=" * 70)
    log.info("Snapshot params: %s", args.params_snapshot)
    log.info("Output: %s", args.output_file)
    log.info("Modelos: %d  |  Datasets: %d  |  Seeds: %d",
             len(MODELS_TO_RUN), len(DATASETS), len(seeds))

    # Verificar snapshot
    if not args.params_snapshot.exists():
        log.error("Snapshot não encontrado: %s", args.params_snapshot)
        log.error("Crie com: cp results/tuning/best_params_tier2_n5000_cpu.json %s",
                  args.params_snapshot)
        sys.exit(1)

    # Carrega params (uma vez só)
    raw_params = json.loads(args.params_snapshot.read_text())
    tuned = {k: v["best_params"] for k, v in raw_params.items() if "best_params" in v}
    log.info("Params carregados: %d combos", len(tuned))

    # Verifica se todos os modelos estão tunados
    missing = []
    for runner_name, variant_name in MODELS_TO_RUN:
        for ds in DATASETS:
            key = f"{variant_name}__{ds}"
            if key not in tuned:
                missing.append(key)
    if missing:
        log.error("Params faltando: %s", missing[:5])
        sys.exit(1)
    log.info("✓ Todos os modelos têm params para todos os datasets")

    # Setup checkpoint
    existing = json.loads(args.output_file.read_text()) if args.output_file.exists() else []
    done_keys = _existing_keys(existing)
    all_results = list(existing)
    log.info("Resumindo: %d runs já no arquivo de saída", len(existing))

    total = len(MODELS_TO_RUN) * len(DATASETS) * len(seeds)
    completed = errors = 0
    t_start = time.perf_counter()

    log.info("── Rodando %d experimentos (N=%d cap) ──", total, N_TOTAL_CAP)

    for runner_name, variant_name in MODELS_TO_RUN:
        for dataset in DATASETS:
            key_p = f"{variant_name}__{dataset}"
            params = dict(tuned[key_p])

            for seed in seeds:
                run_key = f"{variant_name}__{dataset}__{seed}"
                if run_key in done_keys:
                    continue

                result = run_single_experiment(
                    model_name=runner_name,
                    dataset_name=dataset,
                    seed=seed,
                    model_params=params,
                    n_samples_cap=N_TOTAL_CAP,
                )
                result["model_variant"] = variant_name
                result["n_samples_cap"] = N_TOTAL_CAP
                result["script"] = "C_parallel"

                all_results.append(result)
                done_keys.add(run_key)
                completed += 1
                if result["status"] != "ok":
                    errors += 1

                if completed % 5 == 0:
                    _save_json(args.output_file, all_results)

                elapsed = time.perf_counter() - t_start
                eta = (elapsed / completed * (total - completed)) if completed else 0
                f1 = result.get("f1_macro", float("nan"))
                log.info("[%d/%d] %s / %s / seed=%d — %s f1=%.4f | ETA %.0fm",
                         completed, total, variant_name, dataset, seed,
                         result["status"], f1, eta / 60)

    _save_json(args.output_file, all_results)
    log.info("=" * 70)
    log.info("=== Script C concluído: %d/%d ok, %d errors ===",
             completed - errors, total, errors)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
