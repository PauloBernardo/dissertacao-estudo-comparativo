#!/usr/bin/env python3
"""End-to-end benchmark de eficiência para os Transformers tabulares.

Objetivo
--------
Medir custo computacional de uma *run real* no protocolo Tier 2:
    1. carregar dataset real
    2. subamostrar estratificadamente para um N alvo
    3. split 70/30
    4. preprocessar
    5. treinar
    6. predizer no teste

Isso evita o viés do microbenchmark antigo (1 época em dados sintéticos),
mantendo a comparação honesta como benchmark computacional de paradigmas
distintos, não como "ranking único de esparsidade".
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loaders import DatasetLoader
from src.data.preprocessing import make_splits, preprocess
from src.experiments.reproducibility import set_global_seed
from src.metrics.performance import compute_performance
from src.models.ft_transformer_cur_wrapper import FTTransformerCURColnorm
from src.models.ft_transformer_saint_wrapper import SAINTColnorm
from src.models.transformers.ft_transformer import FTTransformer


def _peak_ram_mb() -> float:
    """Return absolute peak RSS of the current worker process in MB."""
    # ru_maxrss is reported in KiB on Linux, which is the target platform here.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _load_params(path: Path, key: str) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        data = json.load(f)
    return dict(data.get(key, {}).get("best_params", {}))


def _resolve_model_params(dataset: str) -> list[tuple[str, dict, object]]:
    gpu_tuning = Path("results/tuning/best_params_tier2_n5000_gpu.json")
    valloss_tuning = Path("results/tuning/best_params_ftcur_saint_valloss.json")

    params_softmax = _load_params(gpu_tuning, f"FTTransformer_softmax__{dataset}")
    params_topk = _load_params(gpu_tuning, f"FTTransformer_topk__{dataset}")
    params_entmax = _load_params(gpu_tuning, f"FTTransformer_entmax__{dataset}")
    params_sparsemax = _load_params(gpu_tuning, f"FTTransformer_sparsemax__{dataset}")
    params_cur = _load_params(valloss_tuning, f"FTTransformerCURColnorm__{dataset}__val_loss")
    params_saint = _load_params(valloss_tuning, f"SAINTColnorm__{dataset}__val_loss")

    params_topk.setdefault("attention_type", "topk")
    params_topk.setdefault("topk_ratio", 0.10)
    params_softmax.setdefault("attention_type", "softmax")
    params_entmax.setdefault("attention_type", "entmax")
    params_entmax.setdefault("alpha", 1.5)
    params_sparsemax.setdefault("attention_type", "sparsemax")
    params_cur.setdefault("early_stop_metric", "val_loss")
    params_saint.setdefault("early_stop_metric", "val_loss")

    return [
        ("FT-Softmax", params_softmax, FTTransformer),
        ("FT-TopK", params_topk, FTTransformer),
        ("FT-Entmax", params_entmax, FTTransformer),
        ("FT-Sparsemax", params_sparsemax, FTTransformer),
        ("FT-CUR", params_cur, FTTransformerCURColnorm),
        ("SAINT", params_saint, SAINTColnorm),
    ]


def _resolve_model_spec(dataset: str, model_name: str) -> tuple[dict, object]:
    for name, params, cls in _resolve_model_params(dataset):
        if name == model_name:
            return params, cls
    raise KeyError(f"Modelo desconhecido para benchmark: {model_name}")


def _subsample(X: np.ndarray, y: np.ndarray, n_total: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(X) <= n_total:
        return X, y
    from sklearn.model_selection import StratifiedShuffleSplit

    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_total, random_state=seed)
    idx_keep, _ = next(sss.split(X, y))
    return X[idx_keep], y[idx_keep]


def _measure_one_run(model_name: str, model_factory, X: np.ndarray, y: np.ndarray, seed: int) -> dict:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    fit_s = predict_s = None
    success = True
    error_message = None
    failure_type = None

    try:
        X_train, X_test, y_train, y_test = make_splits(X, y, test_size=0.30, seed=seed)
        X_train_p, X_test_p, y_train_p, y_test_p = preprocess(
            X_train, X_test, y_train, y_test, label_format="binary"
        )

        model = model_factory()
        fit_start = time.perf_counter()
        model.fit(X_train_p, y_train_p)
        fit_end = time.perf_counter()
        fit_s = fit_end - fit_start

        pred_start = time.perf_counter()
        y_pred = model.predict(X_test_p)
        y_proba = model.predict_proba(X_test_p) if hasattr(model, "predict_proba") else None
        pred_end = time.perf_counter()
        predict_s = pred_end - pred_start
        metrics = compute_performance(y_test_p, y_pred, y_proba)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "cublas" in str(e).lower():
            success = False
            metrics = {}
            error_message = str(e)
            failure_type = "oom"
        else:
            raise

    total_s = time.perf_counter() - t0

    peak_vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    peak_ram = _peak_ram_mb()

    return {
        "model": model_name,
        "n_total": len(X),
        "n_train": int(round(len(X) * 0.70)),
        "n_test": len(X) - int(round(len(X) * 0.70)),
        "total_time_s": total_s,
        "fit_time_s": fit_s,
        "predict_time_s": predict_s,
        "peak_ram_mb": peak_ram,
        "peak_vram_mb": peak_vram,
        "success": success,
        "error_message": error_message,
        "failure_type": failure_type,
        **metrics,
    }


def _measure_in_subprocess(dataset: str, model_name: str, n_total: int, seed: int) -> dict:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-dataset", dataset,
        "--worker-model", model_name,
        "--worker-n-total", str(n_total),
        "--worker-seed", str(seed),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if proc.returncode != 0:
        stderr = proc.stderr.strip() or proc.stdout.strip() or "worker failed"
        failure_type = "worker_error"
        if proc.returncode < 0:
            signal_num = -proc.returncode
            failure_type = f"signal_{signal_num}"
        elif "out of memory" in stderr.lower() or "cublas" in stderr.lower():
            failure_type = "oom"
        return {
            "model": model_name,
            "dataset": dataset,
            "n_total": n_total,
            "seed": seed,
            "success": False,
            "error_message": stderr,
            "failure_type": failure_type,
        }

    return json.loads(proc.stdout)


def _run_worker(dataset: str, model_name: str, n_total: int, seed: int) -> dict:
    set_global_seed(seed)
    X_full, y_full, _ = DatasetLoader.load(dataset)
    X, y = _subsample(X_full, y_full, n_total=n_total, seed=seed)
    params, cls = _resolve_model_spec(dataset, model_name)
    result = _measure_one_run(model_name, lambda: cls(**params), X, y, seed)
    result["dataset"] = dataset
    result["seed"] = seed
    return result


def _load_existing_results(output_file: Path | None) -> list[dict]:
    if output_file is None or not output_file.exists():
        return []
    with output_file.open() as f:
        data = json.load(f)
    return list(data)


def _is_completed_result(result: dict) -> bool:
    if bool(result.get("success")) and result.get("fit_time_s") is not None:
        return True
    return result.get("failure_type") == "oom"


def _save_results(output_file: Path | None, results: list[dict]) -> None:
    if output_file is None:
        return
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, indent=2, default=str))


def _print_header(dataset: str, results_count: int, output_file: Path | None) -> None:
    print(f"\n=== Dataset: {dataset} ===")
    print(
        f"{'Modelo':<15} | {'N':<6} | {'Tempo total (s)':<15} | "
        f"{'Fit (s)':<10} | {'RAM CPU (MB)':<15} | {'VRAM GPU (MB)':<15} | {'F1':<8}"
    )
    print("-" * 108)
    if results_count:
        print(f"Retomando de {results_count} medições já salvas em {output_file}")


def run_benchmark(datasets: list[str], sizes: list[int], seed: int, output_file: Path | None) -> None:
    if not torch.cuda.is_available():
        print("Aviso: CUDA não disponível. O benchmark de VRAM registrará 0MB.")

    set_global_seed(seed)
    results = _load_existing_results(output_file)
    done_keys = {
        (r.get("dataset"), r.get("model"), r.get("n_total"), r.get("seed"))
        for r in results
        if _is_completed_result(r)
    }

    for dataset in datasets:
        models = _resolve_model_params(dataset)
        dataset_results_count = sum(1 for r in results if r.get("dataset") == dataset)
        _print_header(dataset, dataset_results_count, output_file)

        for name, params, cls in models:
            if not params:
                print(f"{name:<15} | {'-':<6} | {'SKIP':<15} | {'-':<10} | {'-':<15} | {'-':<15} | sem params")
                continue

            for n_total in sizes:
                run_key = (dataset, name, n_total, seed)
                if run_key in done_keys:
                    print(f"{name:<15} | {n_total:<6} | {'SKIP':<15} | {'-':<10} | {'-':<15} | {'-':<15} | ja salvo")
                    continue

                result = _measure_in_subprocess(dataset, name, n_total, seed)
                results.append(result)
                if _is_completed_result(result):
                    done_keys.add(run_key)
                _save_results(output_file, results)

                if result["success"]:
                    print(
                        f"{name:<15} | {result['n_total']:<6} | {result['total_time_s']:<15.2f} | "
                        f"{result['fit_time_s']:<10.2f} | {result['peak_ram_mb']:<15.1f} | "
                        f"{result['peak_vram_mb']:<15.1f} | {result['f1_macro']:<8.4f}"
                    )
                else:
                    print(
                        f"{name:<15} | {result['n_total']:<6} | {'OOM':<15} | "
                        f"{'OOM':<10} | {'OOM':<15} | {'OOM':<15} | {'OOM':<8}"
                    )
                    break

    _save_results(output_file, results)
    if output_file is not None:
        print(f"\nSalvo em: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ADULT", "CREDIT", "HIGGS50K"],
        help="Datasets reais usados no benchmark",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[1000, 3000, 5000, 10000, 15000, 20000, 30000, 48842],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("results/benchmark_transformers_efficiency.json"),
    )
    parser.add_argument("--worker-dataset")
    parser.add_argument("--worker-model")
    parser.add_argument("--worker-n-total", type=int)
    parser.add_argument("--worker-seed", type=int)
    args = parser.parse_args()

    if args.worker_dataset is not None:
        result = _run_worker(
            dataset=args.worker_dataset,
            model_name=args.worker_model,
            n_total=args.worker_n_total,
            seed=args.worker_seed,
        )
        print(json.dumps(result))
        return

    run_benchmark(args.datasets, args.sizes, args.seed, args.output_file)


if __name__ == "__main__":
    main()
