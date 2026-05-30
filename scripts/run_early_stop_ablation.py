#!/usr/bin/env python3
"""Ablação: critério de early stopping para FT-CUR e SAINT em imbalanceados.

Testa a hipótese H2 do relatório:
  O colapso do FT-CUR (e potencialmente SAINT) em CREDIT/BANK pode ser causado
  por uso de val_acc como critério de early stopping. Em datasets imbalanceados,
  val_acc favorece checkpoints que acertam majoritariamente a classe dominante,
  mas que ainda têm F1-macro baixo.

Comparação:
  - val_acc       (default original)
  - val_loss      (BCE loss em validação)
  - val_f1_macro  (F1-macro, alinhado à métrica do estudo)

Modelos: FT-CUR (Nyströmformer), SAINT
Datasets: CREDIT, BANK (os 2 onde houve colapso significativo)

Reusa params tunados do tier2_n5000 — varia APENAS o early_stop_metric.

Output: results/tier2_early_stop_ablation.json

Usage:
    python scripts/run_early_stop_ablation.py --seeds 30
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/early_stop_ablation.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

N_TOTAL_CAP = 7143
DATASETS = ["CREDIT", "BANK"]
MODELS = [
    ("FTTransformerCURColnorm", "FTTransformerCURColnorm"),
    ("SAINTColnorm",            "SAINTColnorm"),
]
EARLY_STOP_METRICS = ["val_acc", "val_loss", "val_f1_macro"]


def _save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--output-file", type=Path,
                        default=Path("results/tier2_early_stop_ablation.json"))
    parser.add_argument("--params-file", type=Path,
                        default=Path("results/tuning/best_params_tier2_n5000_snapshot.json"))
    args = parser.parse_args()

    from src.experiments.runner import run_single_experiment

    # Carrega params já tunados
    raw_params = json.loads(args.params_file.read_text())
    tuned = {k: v["best_params"] for k, v in raw_params.items() if "best_params" in v}
    log.info("Params carregados: %d combos", len(tuned))

    # Resume
    existing = json.loads(args.output_file.read_text()) if args.output_file.exists() else []
    done_keys = set()
    for r in existing:
        key = (r.get("model_variant"), r.get("dataset"),
               r.get("seed"), r.get("early_stop_metric", "val_acc"))
        done_keys.add(key)
    log.info("Resumindo: %d runs já feitos", len(existing))

    seeds = list(range(args.seeds))
    total = len(MODELS) * len(DATASETS) * len(seeds) * len(EARLY_STOP_METRICS)
    log.info("=" * 70)
    log.info("Ablação early_stop_metric — FT-CUR e SAINT em CREDIT/BANK")
    log.info("=" * 70)
    log.info("Modelos: %d | Datasets: %d | Métricas: %d | Seeds: %d | Total: %d",
             len(MODELS), len(DATASETS), len(EARLY_STOP_METRICS), len(seeds), total)

    all_results = list(existing)
    completed = errors = 0
    t_start = time.perf_counter()

    for runner_name, variant_name in MODELS:
        for dataset in DATASETS:
            base_params = dict(tuned.get(f"{variant_name}__{dataset}", {}))
            if not base_params:
                log.warning("Sem params tunados para %s/%s — pulando", variant_name, dataset)
                continue

            for metric in EARLY_STOP_METRICS:
                params = dict(base_params)
                params["early_stop_metric"] = metric

                for seed in seeds:
                    key = (variant_name, dataset, seed, metric)
                    if key in done_keys:
                        continue

                    result = run_single_experiment(
                        model_name=runner_name,
                        dataset_name=dataset,
                        seed=seed,
                        model_params=params,
                        n_samples_cap=N_TOTAL_CAP,
                    )
                    result["model_variant"] = variant_name
                    result["early_stop_metric"] = metric
                    result["n_samples_cap"] = N_TOTAL_CAP

                    all_results.append(result)
                    done_keys.add(key)
                    completed += 1
                    if result["status"] != "ok":
                        errors += 1

                    if completed % 5 == 0:
                        _save_json(args.output_file, all_results)

                    elapsed = time.perf_counter() - t_start
                    eta = (elapsed / completed * (total - completed)) if completed else 0
                    f1 = result.get("f1_macro", float("nan"))
                    log.info("[%d/%d] %s/%s/%s/seed=%d — f1=%.4f | ETA %.0fm",
                             completed, total, variant_name, dataset, metric,
                             seed, f1, eta / 60)

    _save_json(args.output_file, all_results)
    log.info("=" * 70)
    log.info("=== Concluído: %d/%d ok, %d errors ===", completed - errors, total, errors)
    log.info("=" * 70)

    # Análise rápida
    print()
    print("=" * 80)
    print("Resumo F1-macro por modelo × dataset × early_stop_metric")
    print("=" * 80)
    import numpy as np
    scores = defaultdict(list)
    for r in all_results:
        if r.get("status") != "ok": continue
        key = (r["model_variant"], r["dataset"], r.get("early_stop_metric", "val_acc"))
        scores[key].append(r.get("f1_macro", float("nan")))

    print(f"\n{'Modelo':<28} {'Dataset':<10} {'Métrica':<15} {'F1-macro':>10}")
    print("─" * 70)
    for (m, d, metric), vals in sorted(scores.items()):
        if vals:
            print(f"{m:<28} {d:<10} {metric:<15} {np.mean(vals):>10.4f}")


if __name__ == "__main__":
    main()
