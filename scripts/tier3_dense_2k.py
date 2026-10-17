"""Controle do Tier 3: LSSVM denso treinado em 2.000 amostras sorteadas do treino.

Mesmo orçamento de pontos que m = 2000, mas descartando o resto dos 200 mil.
Pergunta: o Nyström ganha F1 por enxergar os 200 mil?

Protocolo do Tier 3: seleção única (semente 0) por Optuna TPE, 20 trials, hold-out
80/20 dentro das 2 mil; ajuste final com 10 sementes (cada uma com seu sorteio de
2 mil), teste = os mesmos 50 mil do Tier 3.

    .venv/bin/python scripts/tier3_dense_2k.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_tier2_gridcv as g  # noqa: E402
import run_tier3 as t3  # noqa: E402

N_DENSE = 2000
VARIANT = "StandardLSSVM"


def sample_2k(dataset: str, seed: int, fmt: str):
    X_tr, X_te, y_tr, y_te, _ = t3.load_split(dataset, seed, fmt)
    X_2k, _, y_2k, _ = train_test_split(X_tr, y_tr, train_size=N_DENSE,
                                        stratify=y_tr, random_state=seed)
    return X_2k, X_te, y_2k, y_te


def fit(params: dict, seed: int, X, y):
    pipe, _ = g._build_pipeline(VARIANT, seed)
    pipe.set_params(**{f"clf__{k}": v for k, v in params.items()})
    return pipe.fit(X, y)


def select(dataset: str, fmt: str, n_trials: int) -> dict:
    X_2k, _, y_2k, _ = sample_2k(dataset, t3.SELECT_SEED, fmt)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_2k, y_2k, test_size=t3.VAL_SIZE, stratify=y_2k, random_state=t3.SELECT_SEED)

    def objective(trial):
        params = {"sigma": trial.suggest_categorical("sigma", t3.SIGMA_NYS),
                  "tau": trial.suggest_float("tau", 1e-2, 1e4, log=True)}
        g.set_global_seed(t3.SELECT_SEED)
        return t3._f1(y_val, fit(params, t3.SELECT_SEED, X_fit, y_fit).predict(X_val), fmt)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=t3.SELECT_SEED))
    study.optimize(objective, n_trials=n_trials)
    return {"params": study.best_params, "best_val_f1": study.best_value,
            "n_trials": n_trials, "n_fit": len(X_fit), "n_val": len(X_val)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--datasets", nargs="+", default=t3.DATASETS)
    p.add_argument("--n-trials", type=int, default=t3.N_TRIALS)
    p.add_argument("--output", type=Path, default=ROOT / "results" / "tier3_dense_2k.json")
    a = p.parse_args()

    fmt = g._label_format(VARIANT)
    out = {"protocol": "Tier 3: Optuna TPE 20 trials (semente 0, hold-out 80/20 nas 2 mil); "
                       "final com 10 sementes; teste 50 mil",
           "selection": {}, "final": []}
    for d in a.datasets:
        sel = select(d, fmt, a.n_trials)
        out["selection"][d] = sel
        print(d, "seleção", sel, flush=True)
        for s in a.seeds:
            g.set_global_seed(s)
            X_2k, X_te, y_2k, y_te = sample_2k(d, s, fmt)
            t0 = time.perf_counter()
            pipe = fit(sel["params"], s, X_2k, y_2k)
            fit_s = time.perf_counter() - t0
            rec = {"dataset": d, "seed": s, "n_train": N_DENSE, "n_test": len(X_te),
                   "params": sel["params"], "fit_time_s": fit_s,
                   "test_f1_macro": t3._f1(y_te, pipe.predict(X_te), fmt)}
            out["final"].append(rec)
            print(json.dumps(rec), flush=True)
            a.output.write_text(json.dumps(out, indent=1))
    for d in a.datasets:
        v = [r["test_f1_macro"] for r in out["final"] if r["dataset"] == d]
        print(f"{d}: F1 = {np.mean(v):.4f} ± {np.std(v, ddof=1):.4f} (n={len(v)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
