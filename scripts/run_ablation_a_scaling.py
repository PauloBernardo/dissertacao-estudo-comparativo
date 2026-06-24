#!/usr/bin/env python3
"""Ablação A — Escalabilidade amostral: N=400 → N=2000.

Protocol:
    For each (variant, synthetic_dataset, seed):
      1. Load best_params from tier1_gridcv.json for (variant, dataset_2D, seed).
         dataset_2D maps TWS_2k→TWS, TWM_2k→TWM, TWC_2k→TWC.
      2. Build model with those fixed params (no Grid+CV — params transfer).
      3. Hold-out 70/30 on N=2000, fit on train, evaluate on test.

Saves to results/ablation_a_scaling.json. Resumable.

Usage
-----
    python scripts/run_ablation_a_scaling.py [--output FILE]
                                              [--models M1 M2 ...]
                                              [--seeds 0 1 ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loaders import DatasetLoader
from src.data.preprocessing import _convert_labels
from src.experiments.reproducibility import set_global_seed
from src.experiments.runner import _build_model
from src.metrics.sparsity import transformer_sparsity
from src.tuning.grids import GRIDS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("ablation_a")

ABLATION_DATASETS = ["TWS_2k", "TWM_2k", "TWC_2k"]
TIER1_DATASET_MAP = {"TWS_2k": "TWS", "TWM_2k": "TWM", "TWC_2k": "TWC"}

LSSVM_VARIANTS = {
    "StandardLSSVM", "PCPLSSVm", "FSALSSVm", "IPLSSVm",
    "PruningLSSVM", "OppositeMapsLSSVM",
    "ADMMNesterovLSSVM", "ADMMElasticNet",
    "FISTANesterov", "DualFISTA", "NystromLSSVMColnorm",
}

TRANSFORMER_VARIANTS = {
    "FTTransformer_softmax", "FTTransformer_topk",
    "FTTransformer_entmax", "FTTransformer_sparsemax",
    "FTTransformerCURColnorm", "SAINTColnorm",
}

NON_TRANSFORMER_VARIANTS = LSSVM_VARIANTS | {"XGBoost"}
ALL_VARIANTS = NON_TRANSFORMER_VARIANTS | TRANSFORMER_VARIANTS


def _run_key(variant: str, dataset: str, seed: int) -> str:
    return f"{variant}__{dataset}__seed{seed}"


def _save(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(records, indent=2, default=str))
    tmp.replace(path)


def _load_tier1_params(tier1_path: Path) -> dict[tuple, dict]:
    """Returns {(variant, dataset_2d, seed): best_params}."""
    mapping = {}
    with open(tier1_path) as f:
        records = json.load(f)
    for r in records:
        if r.get("status") == "ok" and r.get("best_params"):
            key = (r["variant"], r["dataset"], r["seed"])
            mapping[key] = r["best_params"]
    return mapping


def run_one(
    variant: str,
    dataset_2k: str,
    seed: int,
    best_params: dict,
) -> dict[str, Any]:
    label_format = "signed" if variant in LSSVM_VARIANTS else "binary"
    is_transformer = variant in TRANSFORMER_VARIANTS
    cfg = GRIDS[variant]

    set_global_seed(seed)

    record: dict[str, Any] = {
        "variant":  variant,
        "model":    cfg["model_name"],
        "dataset":  dataset_2k,
        "seed":     seed,
        "label_format": label_format,
        "best_params": best_params,
        "protocol": "transfer_from_tier1",
    }

    try:
        t0 = time.perf_counter()

        X, y, meta = DatasetLoader.load(dataset_2k)
        y = _convert_labels(y, label_format)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=seed,
        )

        # Build model with fixed best_params (no CV)
        all_params = {**cfg["fixed"], **best_params}
        estimator, _ = _build_model(cfg["model_name"], all_params, label_format=label_format)
        if hasattr(estimator, "set_params") and "random_state" in estimator.get_params():
            estimator.set_params(random_state=seed)

        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        if label_format == "signed":
            yt = ((y_test + 1) // 2).astype(int)
            yp = ((y_pred + 1) // 2).astype(int)
        else:
            yt, yp = y_test, y_pred

        clf = pipeline.named_steps["clf"]
        if is_transformer and hasattr(clf, "predict_proba"):
            try:
                clf.predict_proba(pipeline.named_steps["scaler"].transform(X_test[:32]))
                spar_dict = transformer_sparsity(clf)
                spar = spar_dict.get("attention_sparsity", 0.0)
            except Exception:
                spar = 0.0
            nsv = None
        else:
            spar = float(clf.sparsity_ratio_) if hasattr(clf, "sparsity_ratio_") else 0.0
            nsv  = int(clf.n_support_) if hasattr(clf, "n_support_") else None

        record.update({
            "status":        "ok",
            "n_train":       len(X_train),
            "n_test":        len(X_test),
            "n_features":    X.shape[1],
            "test_f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "test_accuracy": float(accuracy_score(yt, yp)),
            "test_mcc":      float(matthews_corrcoef(yt, yp)),
            "sparsity_ratio": spar,
            "n_support_vectors": nsv,
            "wall_time_s":   time.perf_counter() - t0,
        })

    except Exception as e:
        record["status"] = "error"
        record["error"]  = str(e)
        log.warning("FAILED %s__%s__seed%d: %s", variant, dataset_2k, seed, e)

    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",             default="results/ablation_a_scaling.json")
    parser.add_argument("--tier1",              default="results/tier1_gridcv.json")
    parser.add_argument("--models",             nargs="*", default=sorted(NON_TRANSFORMER_VARIANTS))
    parser.add_argument("--seeds",              nargs="*", type=int, default=list(range(30)))
    parser.add_argument("--transformers-only",  action="store_true",
                        help="Override --models to run only transformer variants")
    args = parser.parse_args()

    out_path   = Path(args.output)
    tier1_path = Path(args.tier1)

    if args.transformers_only:
        args.models = sorted(TRANSFORMER_VARIANTS)

    # Load existing results (resumable)
    records: list[dict] = []
    if out_path.exists():
        records = json.loads(out_path.read_text())
    done = {
        _run_key(r["variant"], r["dataset"], r["seed"])
        for r in records if r.get("status") == "ok"
    }

    # Load Tier 1 best_params
    tier1_params = _load_tier1_params(tier1_path)

    variants = [v for v in args.models if v in GRIDS]
    total = len(variants) * len(ABLATION_DATASETS) * len(args.seeds)
    log.info("Ablação A — %d runs total (%d already done)", total, len(done))

    completed = 0
    for variant in variants:
        for ds_2k in ABLATION_DATASETS:
            ds_2d = TIER1_DATASET_MAP[ds_2k]
            for seed in args.seeds:
                key = _run_key(variant, ds_2k, seed)
                if key in done:
                    completed += 1
                    continue

                # Get best_params from Tier 1 for matching 2D dataset + seed
                params = tier1_params.get((variant, ds_2d, seed))
                if params is None:
                    log.warning("No Tier 1 params for %s / %s / seed%d — skipping", variant, ds_2d, seed)
                    continue

                completed += 1
                log.info("[%d/%d] %s__%s__seed%d", completed, total, variant, ds_2k, seed)
                rec = run_one(variant, ds_2k, seed, params)
                records.append(rec)
                _save(out_path, records)

    log.info("Done. %d records saved to %s", len(records), out_path)


if __name__ == "__main__":
    main()
