#!/usr/bin/env python3
"""Mesma validacao (modelo vs. baseline ingenuo no mesmo k), mas no ponto
de PARCIMONIA do Apendice D (mais esparso, dentro de 2 p.p. de F1 de
validacao do melhor) em vez do ponto padrao (argmax) -- responde: "o fix
do plato (Apendice D) e suficiente pra família ℓ1 passar na validacao?"

Reaproveita scratch/pareto_sweep/sparsity_sweep_results.json (grade
completa por config, Tier1) sem re-treinar os modelos ℓ1 -- so extrai o
ponto de parcimonia (mesma logica de scratch/pareto_sweep/analyze_sweep.py)
e treina o baseline ingenuo no MESMO k, na MESMA seed, no MESMO pool de
treino (X_fit, nao X_train inteiro -- e o pool que o sweep do Apendice D
realmente usou pra treinar cada config).

Resumivel via JSON.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import argparse


def _find_repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / "src" / "models" / "ft_transformer_model.py").exists():
            return parent
    raise RuntimeError("Repo root nao encontrado")


ROOT = _find_repo_root()
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.data.loaders import DatasetLoader
from src.data.preprocessing import _convert_labels, make_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("validation_parsimony")

TOL = 0.02
import os
SWEEP_RESULTS = Path(os.environ.get("SWEEP_RESULTS", ROOT / "scratch" / "pareto_sweep" / "sparsity_sweep_results.json"))

STD_HYPERPARAMS = {
    "BCW": {"sigma": 5.0, "tau": 12.5}, "PID": {"sigma": 5.0, "tau": 2.5},
    "HAB": {"sigma": 1.5, "tau": 2.5}, "VCP": {"sigma": 5.0, "tau": 50.0},
    "GCR": {"sigma": 5.0, "tau": 2.5}, "AUS": {"sigma": 5.0, "tau": 0.5},
    "AI4I": {"sigma": 1.5, "tau": 12.5}, "TWS": {"sigma": 0.5, "tau": 12.5},
    "TWM": {"sigma": 0.15, "tau": 0.1}, "TWC": {"sigma": 0.5, "tau": 12.5},
}


def _key(variant: str, dataset: str, seed: int) -> str:
    return f"{variant}__{dataset}__seed{seed}"


def _load_done(path: Path) -> tuple[list[dict], set[str]]:
    if not path.exists():
        return [], set()
    records = json.loads(path.read_text())
    done = {_key(r["variant"], r["dataset"], r["seed"]) for r in records if r.get("status") == "ok"}
    return records, done


def _save(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(records, indent=2, default=str))
    tmp.replace(path)


def _f1(y_true_signed, y_pred_signed) -> float:
    y_true_e = ((y_true_signed + 1) // 2).astype(int)
    y_pred_e = ((y_pred_signed + 1) // 2).astype(int)
    return float(f1_score(y_true_e, y_pred_e, average="macro", zero_division=0))


def _sample_k(X, y_raw, k: int, seed: int):
    n = len(X)
    n_classes = len(np.unique(y_raw))
    if n - k >= n_classes:
        sss_k = StratifiedShuffleSplit(n_splits=1, train_size=k, random_state=seed)
        idx, _ = next(sss_k.split(X, y_raw))
    else:
        idx = np.random.RandomState(seed).choice(n, size=k, replace=False)
    return idx


def _parsimony_points(sweep_records: list[dict]) -> dict[tuple, dict]:
    """(variant,dataset,seed) -> config de parcimonia (mesma logica do
    Apendice D: dentro de TOL do melhor val_f1, a de maior sparsity_ratio)."""
    from collections import defaultdict
    by_group: dict[tuple, list[dict]] = defaultdict(list)
    for r in sweep_records:
        if r.get("status") != "ok":
            continue
        by_group[(r["variant"], r["dataset"], r["seed"])].append(r)

    out = {}
    for key, rows in by_group.items():
        best_val = max(row["val_f1_macro"] for row in rows)
        band = [row for row in rows if row["val_f1_macro"] >= best_val - TOL]
        parsimony = max(band, key=lambda row: row.get("sparsity_ratio", 0.0))
        out[key] = parsimony
    return out


def _reconstruct_fit_test(dataset: str, seed: int):
    """Replica EXATAMENTE o protocolo de scratch/pareto_sweep/run_sparsity_sweep.py:
    make_splits(70/30) -> StratifiedShuffleSplit(20%) em X_train -> X_fit/X_val.
    Scaler ajustado em X_fit (nao X_train inteiro)."""
    X_full, y_full, _meta = DatasetLoader.load(dataset)
    X_train, X_test, y_train_raw, y_test_raw = make_splits(X_full, y_full, test_size=0.30, seed=seed)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    fit_idx, _val_idx = next(sss.split(X_train, y_train_raw))
    X_fit, y_fit_raw = X_train[fit_idx], y_train_raw[fit_idx]

    scaler = StandardScaler().fit(X_fit)
    X_fit_s = scaler.transform(X_fit)
    X_test_s = scaler.transform(X_test)
    y_fit = _convert_labels(y_fit_raw, "signed")
    y_test = _convert_labels(y_test_raw, "signed")
    return X_fit_s, X_test_s, y_fit_raw, y_fit, y_test


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path(__file__).parent / "validation_parsimony_results.json")
    args = p.parse_args()

    from src.models.lssvm.standard import StandardLSSVM

    sweep_records = json.loads(SWEEP_RESULTS.read_text())
    parsimony = _parsimony_points(sweep_records)
    logger.info("Pontos de parcimonia extraidos: %d", len(parsimony))

    records, done = _load_done(args.output)
    logger.info("Retomando de %s: %d entradas ja registradas.", args.output, len(records))

    cache: dict[tuple, Any] = {}

    for i, (key, row) in enumerate(parsimony.items(), start=1):
        variant, dataset, seed = key
        rkey = _key(variant, dataset, seed)
        if rkey in done:
            continue
        k = row.get("n_support_vectors")
        n_fit_total = None  # preenchido abaixo

        cache_key = (dataset, seed)
        if cache_key not in cache:
            cache[cache_key] = _reconstruct_fit_test(dataset, seed)
            if len(cache) > 15:
                cache.pop(next(iter(cache)))
        X_fit_s, X_test_s, y_fit_raw, y_fit, y_test = cache[cache_key]
        n_fit_total = len(X_fit_s)

        record: dict[str, Any] = {
            "variant": variant, "dataset": dataset, "seed": seed,
            "k": k, "n_fit_total": n_fit_total,
            "f1_parsimony": row["test_f1_macro"],
            "sparsity_ratio_parsimony": row.get("sparsity_ratio"),
        }
        if k is None or k <= 0 or k >= n_fit_total:
            record.update({"status": "skip", "reason": "k invalido (colapso total ou sem compressao)"})
            records.append(record)
            done.add(rkey)
            continue
        try:
            k_idx = _sample_k(X_fit_s, y_fit_raw, k, seed)
            X_k, y_k = X_fit_s[k_idx], y_fit[k_idx]

            t0 = time.perf_counter()
            model = StandardLSSVM(**STD_HYPERPARAMS[dataset])
            model.fit(X_k, y_k)
            fit_time = time.perf_counter() - t0
            f1_naive = _f1(y_test, model.predict(X_test_s))

            record.update({"status": "ok", "f1_naive": f1_naive, "fit_s_naive": fit_time})
        except Exception as exc:
            record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            logger.error("  %s/%s/seed%d/k%d FALHOU: %s", variant, dataset, seed, k, exc)

        records.append(record)
        done.add(rkey)
        if i % 50 == 0:
            _save(args.output, records)
            logger.info("[%d/%d] ...", i, len(parsimony))

    _save(args.output, records)
    logger.info("Concluido. Total: %d -> %s", len(records), args.output)


if __name__ == "__main__":
    main()
