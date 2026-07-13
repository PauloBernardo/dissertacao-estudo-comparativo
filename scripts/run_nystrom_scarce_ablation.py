#!/usr/bin/env python3
"""Ablação de escassez — seleção de landmarks do Nyström-SVM com m FIXO em 5%.

Hipótese: com m/n=30% (protocolo principal) a aproximação de Nyström satura —
até a amostragem aleatória captura bem o espectro do kernel —, o que explicaria
a insensibilidade ao seletor. Num regime ESCASSO (m/n=5% fixo, sem tunar), a
colocação dos poucos landmarks pode passar a importar: aqui testamos se algum
seletor "inteligente" (colnorm, kmeans, opposite) bate o random quando há poucos
landmarks.

Protocolo (idêntico ao Tier 1 de run_tier1_gridcv.py), EXCETO:
    - m_ratio FIXO em 0.05 para os quatro seletores (não entra no grid).
    - grid UNIFORME sigma×gamma (24 combos) para todos os datasets e seletores,
      para uma comparação limpa (sem a heterogeneidade de grid do Tier 1 base).

Esparsidade é 0.95 por construção para os quatro → foco 100% em desempenho.

Uso:
    python scripts/run_nystrom_scarce_ablation.py [--variants ...] [--seeds ...]
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

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loaders import DatasetLoader
from src.data.preprocessing import _convert_labels, make_splits
from src.experiments.reproducibility import set_global_seed
from src.experiments.runner import _build_model
from scripts.run_tier1_gridcv import _collect_sparsity_metrics, _compute_test_metrics

TIER1_DATASETS = ["BCW", "PID", "HAB", "VCP", "GCR", "AUS", "AI4I", "TWS", "TWM", "TWC"]
VARIANTS = ["NystromLSSVMRandom", "NystromLSSVMColnorm",
            "NystromLSSVMKmeans", "NystromLSSVMOpposite"]
DEFAULT_SEEDS = list(range(30))
OUTPUT_FILE = Path("results/tier1_scarce_m05.json")

M_RATIO = 0.05

# ── Configuração por família de modelo ──────────────────────────────────────
# LSSVM-Nyström: grade sigma×gamma (24), rótulos ±1.
LSSVM_GRID = {
    "sigma": [0.1, 0.5, 2.0, 8.0],
    "gamma": [0.1, 1.0, 10.0, 30.0, 50.0, 100.0],
}
# FT-CUR: grade n_layers×n_heads (6) --- m_ratio sai da grade (fica FIXO);
# demais hiperparâmetros idênticos aos do Tier 1 (grids.py). Rótulos {0,1}.
FTCUR_GRID = {"n_layers": [1, 2, 3], "n_heads": [2, 4]}
FTCUR_FIXED = {
    "d_model": 32, "lr": 1e-3, "epochs": 40, "patience": 6,
    "early_stop_metric": "val_loss", "batch_size": 4096,
}
FTCUR_SELECTORS = {
    "FTTransformerCURColnorm": "colnorm",
    "FTTransformerCURRandom": "random",
    "FTTransformerCURKmeans": "kmeans",
    "FTTransformerCUROpposite": "opposite",
}


def _cfg(variant: str):
    """(model_name, fixed_params, grid, label_format) para a variante."""
    if variant in FTCUR_SELECTORS:
        fixed = dict(FTCUR_FIXED, m_ratio=M_RATIO,
                     selection_method=FTCUR_SELECTORS[variant])
        return "FTTransformerCURColnorm", fixed, FTCUR_GRID, "binary"
    return variant, {"m_ratio": M_RATIO}, LSSVM_GRID, "signed"


def _grid_size(variant: str) -> int:
    _, _, grid, _ = _cfg(variant)
    n = 1
    for v in grid.values():
        n *= len(v)
    return n


logger = logging.getLogger("scarce")


def _run_key(variant: str, dataset: str, seed: int) -> str:
    return f"{variant}__{dataset}__seed{seed}"


def _existing(records):
    return {_run_key(r["variant"], r["dataset"], r["seed"])
            for r in records if r.get("status") == "ok"}


def _save(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(records, indent=2, default=str))
    tmp.replace(path)


def _pipeline(variant: str, seed: int):
    model_name, fixed, grid, label_fmt = _cfg(variant)
    estimator, _ = _build_model(model_name, fixed, label_format=label_fmt)
    if hasattr(estimator, "set_params") and "random_state" in estimator.get_params():
        estimator.set_params(random_state=seed)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
    return pipe, {f"clf__{k}": v for k, v in grid.items()}


def run_one(variant: str, dataset: str, seed: int) -> dict[str, Any]:
    set_global_seed(seed)
    model_name, _fixed, _grid, label_fmt = _cfg(variant)
    # FT-CUR precisa de GPU/serial; LSSVM paraleliza os folds.
    n_jobs = 1 if variant in FTCUR_SELECTORS else -1
    rec: dict[str, Any] = {
        "variant": variant, "model": model_name, "dataset": dataset, "seed": seed,
        "label_format": label_fmt, "grid_size": _grid_size(variant),
        "m_ratio_fixed": M_RATIO,
    }
    try:
        X, y, meta = DatasetLoader.load(dataset)
        Xtr, Xte, ytr_raw, yte_raw = make_splits(X, y, test_size=0.30, seed=seed)
        ytr = _convert_labels(ytr_raw, label_fmt)
        yte = _convert_labels(yte_raw, label_fmt)
        rec.update({"n_train": int(len(Xtr)), "n_test": int(len(Xte)),
                    "n_features": int(X.shape[1]), "dataset_tier": meta.get("tier")})

        pipe, grid = _pipeline(variant, seed)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        search = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", refit=True,
                              n_jobs=n_jobs, error_score=0.0)
        t0 = time.perf_counter()
        search.fit(Xtr, ytr)
        fit_time = time.perf_counter() - t0

        test_metrics = _compute_test_metrics(search.best_estimator_, Xte, yte, label_fmt)
        clf = search.best_estimator_.named_steps["clf"]
        sparsity = _collect_sparsity_metrics(clf, Xte)
        best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        rec.update({"status": "ok", "best_params": best_params,
                    "cv_score_f1_macro": float(search.best_score_),
                    "fit_time_s": fit_time, **test_metrics, **sparsity})
    except Exception as exc:
        rec.update({"status": "error", "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()})
    return rec


def main() -> int:
    global M_RATIO
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--variants", nargs="+", default=VARIANTS)
    p.add_argument("--datasets", nargs="+", default=TIER1_DATASETS)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--m-ratio", type=float, default=M_RATIO,
                   help="Fração fixa de landmarks (default 0.05).")
    p.add_argument("--output", type=Path, default=OUTPUT_FILE)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    M_RATIO = args.m_ratio

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    logger.setLevel(args.log_level)

    records = json.loads(args.output.read_text()) if args.output.exists() else []
    done = _existing(records)
    plan = [(v, d, s) for v in args.variants for d in args.datasets for s in args.seeds]
    logger.info("Resuming: %d/%d done.", len(done), len(plan))

    interrupted = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: interrupted.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: interrupted.update(flag=True))

    for i, (variant, dataset, seed) in enumerate(plan, 1):
        if interrupted["flag"]:
            break
        if _run_key(variant, dataset, seed) in done:
            continue
        logger.info("[%d/%d] %s %s seed=%d", i, len(plan), variant, dataset, seed)
        rec = run_one(variant, dataset, seed)
        records.append(rec)
        _save(args.output, records)
        if rec["status"] == "ok":
            logger.info("    F1=%.4f acc=%.4f spars=%.3f %.1fs", rec["test_f1_macro"],
                        rec["test_accuracy"], rec.get("sparsity_ratio", float("nan")),
                        rec["fit_time_s"])
        else:
            logger.warning("    ERROR: %s", rec["error"])

    ok = sum(1 for r in records if r.get("status") == "ok")
    logger.info("Done. %d ok / %d.", ok, len(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
