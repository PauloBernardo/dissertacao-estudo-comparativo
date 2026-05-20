"""Bayesian hyperparameter tuning via Optuna.

Reads search spaces from config/models.yaml and runs cross-validated
optimisation for each model × dataset pair.

Usage
-----
    from src.tuning.bayesian import tune_model

    best_params = tune_model(
        model_name="StandardLSSVM",
        dataset_name="BCW",
        n_trials=100,
        cv_folds=5,
        metric="f1_macro",
        seed=0,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import yaml
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold

from src.data.loaders import DatasetLoader
from src.data.preprocessing import preprocess
from src.experiments.reproducibility import set_global_seed

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "models.yaml"

_LSSVM_MODELS = {
    "StandardLSSVM", "ADMMNesterovLSSVM", "PCPLSSVm", "FSALSSVm",
    "PruningLSSVM", "IPLSSVm", "OppositeMapsLSSVM",
}

# Map from config model key → (class_name, module_path)
_MODEL_KEY_MAP = {
    "lssvm_standard":       "StandardLSSVM",
    "lssvm_admm_nesterov":  "ADMMNesterovLSSVM",
    "lssvm_pcp":            "PCPLSSVm",
    "lssvm_fsa":            "FSALSSVm",
    "lssvm_pruning":        "PruningLSSVM",
    "lssvm_ip":             "IPLSSVm",
    "lssvm_opposite_maps":  "OppositeMapsLSSVM",
    "ft_transformer":         "FTTransformer",
    "ft_transformer_topk":    "FTTransformer",
    "ft_transformer_entmax":  "FTTransformer",
    "ft_transformer_sparsemax": "FTTransformer",
}


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _suggest_params(trial: optuna.Trial, search_space: dict) -> dict:
    """Sample hyperparameters from the Optuna trial."""
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        kind = spec["type"]
        if kind == "log_uniform":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
        elif kind == "uniform":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"])
        elif kind == "int_uniform":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown param type: {kind!r}")
    return params


def _build_model(class_name: str, params: dict):
    """Instantiate a model by class name with given params."""
    if class_name == "StandardLSSVM":
        from src.models.lssvm.standard import StandardLSSVM
        return StandardLSSVM(**params)
    elif class_name == "ADMMNesterovLSSVM":
        from src.models.lssvm.primal.admm_nesterov import ADMMNesterovLSSVM
        return ADMMNesterovLSSVM(**params)
    elif class_name == "PCPLSSVm":
        from src.models.lssvm.primal.pcp_lssvm import PCPLSSVm
        return PCPLSSVm(**params)
    elif class_name == "FSALSSVm":
        from src.models.lssvm.primal.fsa_lssvm import FSALSSVm
        return FSALSSVm(**params)
    elif class_name == "PruningLSSVM":
        from src.models.lssvm.dual.p_lssvm import PruningLSSVM
        return PruningLSSVM(**params)
    elif class_name == "IPLSSVm":
        from src.models.lssvm.dual.ip_lssvm import IPLSSVm
        return IPLSSVm(**params)
    elif class_name == "OppositeMapsLSSVM":
        from src.models.lssvm.dual.opposite_maps import OppositeMapsLSSVM
        return OppositeMapsLSSVM(**params)
    elif class_name == "FTTransformer":
        from src.models.transformers.ft_transformer import FTTransformer
        return FTTransformer(**params)
    else:
        raise ValueError(f"Unknown class: {class_name!r}")


def _cv_score(
    class_name: str,
    params: dict,
    X: NDArray,
    y: NDArray,
    cv: StratifiedKFold,
    metric: str,
    label_format: str,
) -> float:
    """Return mean CV score for a given param set."""
    from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

    scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        X_tr_p, X_val_p, y_tr_p, y_val_p = preprocess(
            X_tr, X_val, y_tr, y_val, label_format=label_format
        )

        model = _build_model(class_name, params)
        try:
            model.fit(X_tr_p, y_tr_p)
            y_pred = model.predict(X_val_p)
        except Exception as exc:
            logger.debug("CV fold failed: %s", exc)
            return float("-inf")

        # Convert signed labels back for metric computation
        if label_format == "signed":
            y_val_eval = ((y_val_p + 1) // 2).astype(int)
            y_pred_eval = ((y_pred + 1) // 2).astype(int)
        else:
            y_val_eval = y_val_p
            y_pred_eval = y_pred

        if metric == "f1_macro":
            s = f1_score(y_val_eval, y_pred_eval, average="macro", zero_division=0)
        elif metric == "accuracy":
            s = accuracy_score(y_val_eval, y_pred_eval)
        elif metric == "auc_roc":
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X_val_p)
                    s = roc_auc_score(y_val_eval, proba[:, 1])
                except Exception:
                    s = 0.5
            else:
                s = 0.5
        else:
            raise ValueError(f"Unknown metric: {metric!r}")

        scores.append(s)

    return float(np.mean(scores))


def tune_model(
    model_name: str,
    dataset_name: str,
    n_trials: int = 100,
    cv_folds: int = 5,
    metric: str = "f1_macro",
    seed: int = 0,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Tune a model on a dataset and return the best hyperparameters.

    Parameters
    ----------
    model_name : class name (e.g. "StandardLSSVM") or config key
                 (e.g. "lssvm_standard")
    dataset_name : registered DatasetLoader name
    n_trials : number of Optuna trials
    cv_folds : number of CV folds
    metric : optimisation objective ("f1_macro", "accuracy", "auc_roc")
    seed : global random seed
    timeout : seconds budget (None = unlimited)

    Returns
    -------
    dict with keys: best_params, best_value, n_trials, model, dataset
    """
    set_global_seed(seed)
    config = _load_config()["models"]

    # Resolve model_name → config key + class name
    # Accept both "StandardLSSVM" and "lssvm_standard"
    config_key = None
    class_name = None
    for key, cname in _MODEL_KEY_MAP.items():
        if model_name in (key, cname):
            config_key = key
            class_name = cname
            break

    if config_key is None or config_key not in config:
        raise ValueError(
            f"Model {model_name!r} not found in config/models.yaml. "
            f"Known models: {list(config.keys())}"
        )

    model_cfg = config[config_key]
    search_space = model_cfg.get("search_space", {})
    fixed_params = model_cfg.get("fixed_params", {})

    X, y, _ = DatasetLoader.load(dataset_name)
    label_format = "signed" if class_name in _LSSVM_MODELS else "binary"
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        sampled = _suggest_params(trial, search_space)
        params = {**fixed_params, **sampled}
        # Pass seed to models that support it
        if "random_state" in _build_model.__code__.co_varnames:
            params.setdefault("random_state", seed)
        return _cv_score(class_name, params, X, y, cv, metric, label_format)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best = {**fixed_params, **study.best_params}
    logger.info(
        "Tuning %s / %s: best %s=%.4f (trial %d/%d)",
        model_name, dataset_name, metric, study.best_value,
        study.best_trial.number + 1, n_trials,
    )

    return {
        "model": model_name,
        "dataset": dataset_name,
        "metric": metric,
        "best_params": best,
        "best_value": float(study.best_value),
        "n_trials_completed": len(study.trials),
    }


def tune_all(
    model_names: list[str],
    dataset_names: list[str],
    n_trials_map: dict[int, int] | None = None,
    cv_folds: int = 5,
    metric: str = "f1_macro",
    seed: int = 0,
) -> dict[tuple[str, str], dict]:
    """Tune all model × dataset combinations.

    Parameters
    ----------
    model_names : list of model class names
    dataset_names : list of dataset names
    n_trials_map : {tier: n_trials} — defaults to {1: 150, 2: 100, 3: 50}
    cv_folds : CV folds
    metric : optimisation metric
    seed : random seed

    Returns
    -------
    dict keyed by (model_name, dataset_name) → tune_model result
    """
    from src.data.loaders import DatasetLoader

    tier_map = {
        "HAB": 1, "PID": 1, "BCW": 1, "VCP": 1, "GCR": 1, "AUS": 1,
        "TWS": 1, "TWM": 1, "TWC": 1,
        "ADULT": 2, "BANK": 2, "CREDIT": 2, "TELCO": 2, "SHOPPERS": 2, "HIGGS50K": 2,
        "HIGGS500K": 3, "COVER": 3, "KDD99": 3,
    }
    default_trials = {1: 150, 2: 100, 3: 50}
    n_trials_map = n_trials_map or default_trials

    results = {}
    for model_name in model_names:
        for dataset_name in dataset_names:
            tier = tier_map.get(dataset_name, 1)
            n_trials = n_trials_map.get(tier, 100)
            logger.info("Tuning %s / %s (%d trials)...", model_name, dataset_name, n_trials)
            try:
                results[(model_name, dataset_name)] = tune_model(
                    model_name, dataset_name,
                    n_trials=n_trials, cv_folds=cv_folds,
                    metric=metric, seed=seed,
                )
            except Exception as exc:
                logger.error("Tuning failed: %s / %s — %s", model_name, dataset_name, exc)
                results[(model_name, dataset_name)] = {
                    "model": model_name, "dataset": dataset_name,
                    "status": "error", "error": str(exc),
                }
    return results
