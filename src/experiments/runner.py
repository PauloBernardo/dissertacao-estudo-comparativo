"""Experiment runner orchestrating the 12 × 18 × 30 comparative study.

Each experiment run:
    1. Loads a dataset
    2. Splits into train/test (stratified, with a given seed)
    3. Preprocesses (StandardScaler fit on train only)
    4. Optionally tunes hyperparameters via Optuna
    5. Fits the model and records train time
    6. Evaluates on the test set
    7. Collects sparsity and efficiency metrics
    8. Returns a results dict

Usage
-----
    from src.experiments.runner import run_single_experiment, run_all

    result = run_single_experiment(
        model_name="StandardLSSVM",
        dataset_name="BCW",
        seed=0,
        model_params={"sigma": 3.0, "tau": 1.0},
    )
"""

from __future__ import annotations

import logging
import traceback
from typing import Any

import numpy as np

from src.data.loaders import DatasetLoader
from src.data.preprocessing import make_splits, preprocess
from src.experiments.reproducibility import set_global_seed
from src.metrics.efficiency import measure_fit_time, measure_predict_time, efficiency_metrics
from src.metrics.performance import compute_performance
from src.metrics.sparsity import lssvm_sparsity, transformer_sparsity

logger = logging.getLogger(__name__)

# ── Model registry ─────────────────────────────────────────────────────────────

def _build_model(model_name: str, model_params: dict[str, Any], label_format: str):
    """Instantiate a model by name."""
    if model_name == "StandardLSSVM":
        from src.models.lssvm.standard import StandardLSSVM
        return StandardLSSVM(**model_params), label_format
    elif model_name == "ADMMNesterovLSSVM":
        from src.models.lssvm.primal.admm_nesterov import ADMMNesterovLSSVM
        return ADMMNesterovLSSVM(**model_params), label_format
    elif model_name == "FISTANesterovLSSVM":
        from src.models.lssvm.primal.fista_lssvm import FISTANesterovLSSVM
        return FISTANesterovLSSVM(**model_params), label_format
    elif model_name == "DualFISTALSSVM":
        from src.models.lssvm.dual.fista_dual_lssvm import DualFISTALSSVM
        return DualFISTALSSVM(**model_params), label_format
    elif model_name == "PCPLSSVm":
        from src.models.lssvm.primal.pcp_lssvm import PCPLSSVm
        return PCPLSSVm(**model_params), label_format
    elif model_name == "FSALSSVm":
        from src.models.lssvm.primal.fsa_lssvm import FSALSSVm
        return FSALSSVm(**model_params), label_format
    elif model_name == "PruningLSSVM":
        from src.models.lssvm.dual.p_lssvm import PruningLSSVM
        return PruningLSSVM(**model_params), label_format
    elif model_name == "IPLSSVm":
        from src.models.lssvm.dual.ip_lssvm import IPLSSVm
        return IPLSSVm(**model_params), label_format
    elif model_name == "OppositeMapsLSSVM":
        from src.models.lssvm.dual.opposite_maps import OppositeMapsLSSVM
        return OppositeMapsLSSVM(**model_params), label_format
    elif model_name == "FTTransformer":
        from src.models.transformers.ft_transformer import FTTransformer
        return FTTransformer(**model_params), "binary"
    elif model_name == "NystromLSSVMColnorm":
        from src.models.nystrom_lssvm_wrapper import NystromLSSVMColnorm
        return NystromLSSVMColnorm(**model_params), "signed"
    elif model_name == "FTTransformerCURColnorm":
        from src.models.ft_transformer_cur_wrapper import FTTransformerCURColnorm
        return FTTransformerCURColnorm(**model_params), "binary"
    elif model_name == "SAINTColnorm":
        from src.models.ft_transformer_saint_wrapper import SAINTColnorm
        return SAINTColnorm(**model_params), "binary"
    elif model_name == "XGBoost":
        from src.models.xgboost_wrapper import XGBoostBaseline
        return XGBoostBaseline(**model_params), "binary"
    else:
        raise ValueError(f"Unknown model: {model_name!r}")


_LSSVM_MODELS = {
    "StandardLSSVM", "ADMMNesterovLSSVM",
    "FISTANesterovLSSVM", "DualFISTALSSVM", "PCPLSSVm", "FSALSSVm",
    "PruningLSSVM", "IPLSSVm", "OppositeMapsLSSVM", "NystromLSSVMColnorm",
}
_TRANSFORMER_MODELS = {"FTTransformer"}
# Inter-instance models expose n_support_/sparsity_ratio_/n_samples_fit_ like LSSVMs
_INTER_INSTANCE_MODELS = {"FTTransformerCURColnorm", "SAINTColnorm"}


def _collect_sparsity(model_name: str, model, X_test_proc) -> dict[str, float]:
    if model_name in _LSSVM_MODELS or model_name in _INTER_INSTANCE_MODELS:
        return lssvm_sparsity(model)
    elif model_name in _TRANSFORMER_MODELS:
        # Run a small forward pass to populate _last_attn_weights
        try:
            model.predict_proba(X_test_proc[:min(32, len(X_test_proc))])
            return transformer_sparsity(model)
        except Exception:
            return {}
    return {}


# ── Single run ─────────────────────────────────────────────────────────────────

def run_single_experiment(
    model_name: str,
    dataset_name: str,
    seed: int,
    model_params: dict[str, Any] | None = None,
    test_size: float = 0.30,
    n_samples_cap: int | None = None,
    balance_train: bool = False,
) -> dict[str, Any]:
    """Run one experiment and return a results dict.

    Parameters
    ----------
    model_name : one of the registered model names
    dataset_name : registered DatasetLoader name
    seed : random seed for the split and model
    model_params : hyperparameters passed to the model constructor
    test_size : fraction of data held out for testing
    n_samples_cap : if set, stratified subsample to this many rows
                    (deterministic per seed) before train/test split

    Returns
    -------
    dict with all metrics, plus status ("ok" or "error")
    """
    model_params = model_params or {}
    result: dict[str, Any] = {
        "model": model_name,
        "dataset": dataset_name,
        "seed": seed,
        "status": "ok",
    }

    try:
        set_global_seed(seed)

        X, y, meta = DatasetLoader.load(dataset_name)

        # Optional: stratified subsample to cap N (for Tier 2 N=5000 protocol)
        if n_samples_cap is not None and len(X) > n_samples_cap:
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(
                n_splits=1, train_size=n_samples_cap, random_state=seed)
            idx_keep, _ = next(sss.split(X, y))
            X, y = X[idx_keep], y[idx_keep]

        # Determine label format: LSSVM models need signed {-1,+1}
        label_format = "signed" if model_name in _LSSVM_MODELS else "binary"

        X_train, X_test, y_train, y_test = make_splits(
            X, y, test_size=test_size, seed=seed
        )
        X_train_p, X_test_p, y_train_p, y_test_p = preprocess(
            X_train, X_test, y_train, y_test, label_format=label_format
        )

        # Optional: balance training set by undersampling majority class.
        # Test set is preserved as-is (original class distribution),
        # ensuring comparability of test metric with the imbalanced protocol.
        if balance_train:
            classes = np.unique(y_train_p)
            if len(classes) == 2:
                c1, c2 = classes
                idx1 = np.where(y_train_p == c1)[0]
                idx2 = np.where(y_train_p == c2)[0]
                n_minor = min(len(idx1), len(idx2))
                rng = np.random.RandomState(seed)
                if len(idx1) > n_minor:
                    idx1 = rng.choice(idx1, size=n_minor, replace=False)
                if len(idx2) > n_minor:
                    idx2 = rng.choice(idx2, size=n_minor, replace=False)
                idx = np.sort(np.concatenate([idx1, idx2]))
                X_train_p = X_train_p[idx]
                y_train_p = y_train_p[idx]

        model, _ = _build_model(model_name, model_params, label_format)
        if hasattr(model, "random_state"):
            if hasattr(model, "set_params"):
                model.set_params(random_state=seed)
            else:
                model.random_state = seed

        # Fit with timing
        model, train_time = measure_fit_time(model, X_train_p, y_train_p)

        # Predict with timing
        y_pred, predict_time = measure_predict_time(model, X_test_p)

        # Performance (always compute proba if available)
        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test_p)
                # Convert signed labels back to {0,1} for sklearn metrics
                if label_format == "signed":
                    y_test_eval = ((y_test_p + 1) // 2).astype(int)
                    y_pred_eval = ((y_pred + 1) // 2).astype(int)
                else:
                    y_test_eval = y_test_p
                    y_pred_eval = y_pred
            except Exception:
                y_test_eval = y_test_p
                y_pred_eval = y_pred
        else:
            if label_format == "signed":
                y_test_eval = ((y_test_p + 1) // 2).astype(int)
                y_pred_eval = ((y_pred + 1) // 2).astype(int)
            else:
                y_test_eval = y_test_p
                y_pred_eval = y_pred

        perf = compute_performance(y_test_eval, y_pred_eval, y_proba)
        result.update(perf)

        # Efficiency
        eff = efficiency_metrics(
            train_time, predict_time, len(X_train_p), len(X_test_p)
        )
        result.update(eff)

        # Sparsity
        sparsity = _collect_sparsity(model_name, model, X_test_p)
        result.update(sparsity)

        # Dataset metadata
        result.update({
            "n_train": len(X_train_p),
            "n_test": len(X_test_p),
            "n_features": X_train_p.shape[1],
            "dataset_tier": meta.get("tier", "?"),
        })

    except Exception as exc:
        result["status"] = "error"
        result["error_message"] = str(exc)
        result["traceback"] = traceback.format_exc()
        logger.error(
            "Experiment failed: model=%s dataset=%s seed=%d — %s",
            model_name, dataset_name, seed, exc,
        )

    return result


# ── Batch runner ───────────────────────────────────────────────────────────────

def run_all(
    model_names: list[str],
    dataset_names: list[str],
    seeds: list[int],
    model_params_map: dict[str, dict] | None = None,
    test_size: float = 0.30,
    n_jobs: int = 1,
) -> list[dict[str, Any]]:
    """Run all combinations of models × datasets × seeds.

    Parameters
    ----------
    model_names : list of model names to evaluate
    dataset_names : list of dataset names
    seeds : list of random seeds
    model_params_map : {model_name: {param: value, ...}}
    test_size : test fraction
    n_jobs : 1 = sequential; -1 = all CPUs via joblib

    Returns
    -------
    list of result dicts (one per combination)
    """
    model_params_map = model_params_map or {}
    combos = [
        (m, d, s)
        for m in model_names
        for d in dataset_names
        for s in seeds
    ]

    def _run(args):
        m, d, s = args
        params = model_params_map.get(m, {})
        logger.info("Running %s / %s / seed=%d", m, d, s)
        return run_single_experiment(m, d, s, model_params=params, test_size=test_size)

    if n_jobs == 1:
        return [_run(c) for c in combos]

    try:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs)(
            delayed(run_single_experiment)(
                m, d, s,
                model_params=model_params_map.get(m, {}),
                test_size=test_size,
            )
            for m, d, s in combos
        )
        return list(results)
    except ImportError:
        logger.warning("joblib not available, falling back to sequential")
        return [_run(c) for c in combos]
