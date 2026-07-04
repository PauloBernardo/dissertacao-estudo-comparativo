#!/usr/bin/env python3
"""Ablação Tier 2 em N maior com hiperparâmetros FIXOS (sem GridSearchCV).

Mesmo protocolo, datasets e roster de 15 modelos do Tier 2 (`run_tier2_gridcv.py`),
mas em vez de re-tunar por GridSearchCV a cada seed, usa os hiperparâmetros
FIXOS extraídos da moda do GridCV de N=2000 (`config/tier2_fixed_params.json`,
gerado por `extract_tier2_fixed_params.py`). Isola o efeito do tamanho de treino
$N$: mesmos modelos, mesmos datasets, params congelados.

Reutiliza (sem modificar) os helpers de `run_tier2_gridcv.py`: carga de dados,
subsampling estratificado, split 70/30, conversão de rótulos, métricas de teste
e coleta de esparsidade.

Uso (CPU — LSSVMs + XGBoost):
    python scripts/run_tier2_fixedparams.py --n-train 5000 \
        --models StandardLSSVM DualFISTA PCPLSSVm FSALSSVm IPLSSVm \
                 NystromLSSVMColnorm ADMMNystromLSSVM FISTANystrom XGBoost

Uso (GPU — Transformers):
    python scripts/run_tier2_fixedparams.py --n-train 5000 \
        --models FTTransformer_softmax FTTransformer_topk FTTransformer_entmax \
                 FTTransformer_sparsemax SAINTColnorm FTTransformerCURColnorm
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Reuso dos helpers do runner de GridCV (guardado por __main__, importar é seguro).
import run_tier2_gridcv as g

DEFAULT_N_TRAIN = 5000
DEFAULT_SEEDS   = list(range(30))
DEFAULT_CONFIG  = ROOT / "config" / "tier2_fixed_params.json"
OUTPUT_FILE     = Path("results/tier2_fixedparams_n5000.json")

logger = logging.getLogger("tier2_fixed")


# ── Pipeline com params fixos ─────────────────────────────────────────────────

def _build_fixed_pipeline(variant: str, dataset: str, seed: int,
                          config: dict) -> tuple[Any, dict]:
    """Pipeline pronto p/ fit: params fixos de GRIDS + moda do GridCV (por dataset)."""
    if variant not in config or dataset not in config[variant]:
        raise KeyError(f"Sem params fixos para ({variant}, {dataset}) em config")
    # Reaproveita a construção do estimador (fixed de GRIDS + scaler + random_state)
    pipeline, _grid = g._build_pipeline(variant, seed)
    modal = config[variant][dataset]
    pipeline.set_params(**{f"clf__{k}": v for k, v in modal.items()})
    return pipeline, modal


# ── Experimento único ─────────────────────────────────────────────────────────

def run_one_fixed(variant: str, dataset: str, seed: int, n_train: int,
                  config: dict) -> dict[str, Any]:
    cfg          = g.GRIDS[variant]
    label_format = g._label_format(variant)
    n_total_cap  = round(n_train / 0.70)

    g.set_global_seed(seed)

    record: dict[str, Any] = {
        "variant":        variant,
        "model":          cfg["model_name"],
        "dataset":        dataset,
        "seed":           seed,
        "label_format":   label_format,
        "n_train_target": n_train,
        "tuning":         "fixed_from_gridcv_n2000",
    }

    try:
        X_full, y_full, meta = g.DatasetLoader.load(dataset)
        X_sub, y_sub = g._subsample(X_full, y_full, n_total_cap, seed)

        X_train, X_test, y_train_raw, y_test_raw = g.make_splits(
            X_sub, y_sub, test_size=0.30, seed=seed)

        y_train = g._convert_labels(y_train_raw, label_format)
        y_test  = g._convert_labels(y_test_raw,  label_format)

        record.update({
            "n_total_sub":  int(len(X_sub)),
            "n_train":      int(len(X_train)),
            "n_test":       int(len(X_test)),
            "n_features":   int(X_full.shape[1]),
            "dataset_tier": meta.get("tier"),
        })

        pipeline, fixed_params = _build_fixed_pipeline(variant, dataset, seed, config)

        t0 = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        test_metrics = g._compute_test_metrics(pipeline, X_test, y_test, label_format)
        predict_time = time.perf_counter() - t0

        clf      = pipeline.named_steps["clf"]
        sparsity = g._collect_sparsity(clf, X_test)

        record.update({
            "status":         "ok",
            "fixed_params":   fixed_params,
            "fit_time_s":     fit_time,
            "predict_time_s": predict_time,
            **test_metrics,
            **sparsity,
        })

    except Exception as exc:
        record.update({
            "status":    "error",
            "error":     f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })

    return record


# ── Orquestrador ──────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n-train",  type=int, default=DEFAULT_N_TRAIN,
                   help="N de treino alvo para TODOS os modelos (default: 5000)")
    p.add_argument("--models",   nargs="+", default=g.DEFAULT_VARIANTS)
    p.add_argument("--datasets", nargs="+", default=g.TIER2_DATASETS)
    p.add_argument("--seeds",    nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--config",   type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output",   type=Path, default=OUTPUT_FILE)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.setLevel(args.log_level)

    config = json.loads(args.config.read_text())

    records       = json.loads(args.output.read_text()) if args.output.exists() else []
    done          = g._existing_keys(records)
    total_planned = len(args.models) * len(args.datasets) * len(args.seeds)
    logger.info("Resumindo: %d/%d entradas já completas.", len(done), total_planned)
    logger.info("Protocolo: params FIXOS, N_train=%d, cap_total=%d",
                args.n_train, round(args.n_train / 0.70))

    plan = [(m, d, s) for m in args.models for d in args.datasets for s in args.seeds]

    interrupted = {"flag": False}
    def _on_signal(signum, _frame):
        logger.warning("Sinal %d — finalizando após entrada atual.", signum)
        interrupted["flag"] = True
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT,  _on_signal)

    t_start = time.perf_counter()
    for i, (variant, dataset, seed) in enumerate(plan, start=1):
        if interrupted["flag"]:
            logger.warning("Interrompido; %d restantes (resumível).", len(plan) - i + 1)
            break

        key = g._run_key(variant, dataset, seed)
        if key in done:
            logger.debug("[%d/%d] SKIP %s", i, len(plan), key)
            continue

        logger.info("[%d/%d] %s", i, len(plan), key)

        t0  = time.perf_counter()
        rec = run_one_fixed(variant, dataset, seed, args.n_train, config)
        rec["wall_time_s"] = time.perf_counter() - t0

        records.append(rec)
        g._save(args.output, records)

        if rec.get("status") == "ok":
            logger.info("    OK  test=%.4f  fit=%.1fs  n_train=%d",
                        rec["test_f1_macro"], rec["fit_time_s"], rec["n_train"])
        else:
            logger.warning("    FAIL  %s", rec.get("error", "?")[:120])

    elapsed = time.perf_counter() - t_start
    n_ok = sum(1 for r in records if r.get("status") == "ok")
    logger.info("=== Concluído. %.1f min. OK=%d/%d. %s ===",
                elapsed / 60, n_ok, total_planned, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
