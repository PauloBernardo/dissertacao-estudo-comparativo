#!/usr/bin/env python3
"""Sensibilidade a ν COM nova seleção: para cada ν, refaz o GridSearchCV completo do Cenário 1
(grade atual do código — ver nota —, mesmos 5 folds estratificados, mesma partição 70/30 por semente) dos primais ℓ1
densos. Complementa study_tikhonov_sensitivity.py, que fixa σ, τ, λ e varia só ν.

    .venv/bin/python scripts/study_tikhonov_retune.py --datasets VCP HAB TWM TWS TWC BCW AUS PID GCR AI4I \
        --seeds 0 1 2 3 4 5 6 7 8 9   # ~4 h em 8 núcleos; retoma do JSON se interrompido

Nota: usa a grade ATUAL de src/tuning/grids.py. Para o ADMM-Nesterov ela tem 75 configurações
(λ inclui 0,5 e 0,001); o Cenário 1 publicado usou 45 (λ ∈ {1; 0,1; 0,01}).
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402
from src.data.loaders import DatasetLoader  # noqa: E402
from src.data.preprocessing import _convert_labels, make_splits  # noqa: E402
from run_tier1_gridcv import _build_pipeline, _compute_test_metrics  # noqa: E402

VARIANTS = ["ADMMNesterovLSSVM", "FISTANesterov"]
NUS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["VCP", "HAB", "TWM", "TWS", "TWC"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--output", default=str(ROOT / "results/study_tikhonov_retune.json"))
    a = ap.parse_args()
    out = Path(a.output)
    recs = json.loads(out.read_text()) if out.exists() else []
    done = {(r["variant"], r["dataset"], r["seed"], r["nu"]) for r in recs}
    for ds in a.datasets:
        X, y, _ = DatasetLoader.load(ds)
        for seed in a.seeds:
            Xtr, Xte, ytr_raw, yte_raw = make_splits(X, y, test_size=0.30, seed=seed)
            ytr, yte = _convert_labels(ytr_raw, "signed"), _convert_labels(yte_raw, "signed")
            for var in VARIANTS:
                for nu in NUS:
                    if (var, ds, seed, nu) in done:
                        continue
                    pipe, grid = _build_pipeline(var, seed)
                    pipe.set_params(clf__tikhonov_reg=nu)
                    search = GridSearchCV(pipe, grid, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                                          scoring="f1_macro", refit=True, n_jobs=-1, error_score=0.0)
                    t0 = time.perf_counter(); search.fit(Xtr, ytr); ft = time.perf_counter() - t0
                    m = _compute_test_metrics(search.best_estimator_, Xte, yte, "signed")
                    clf = search.best_estimator_.named_steps["clf"]
                    recs.append({"variant": var, "dataset": ds, "seed": seed, "nu": nu,
                                 "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
                                 "cv_f1": float(search.best_score_), "test_f1_macro": m["test_f1_macro"],
                                 "sparsity_ratio": float(getattr(clf, "sparsity_ratio_", float("nan"))),
                                 "fit_s": ft})
                    out.write_text(json.dumps(recs, indent=1))
                r = [x for x in recs if x["variant"] == var and x["dataset"] == ds and x["seed"] == seed]
                print(f"{ds:4s} {seed} {var:18s} " + "  ".join(
                    f"ν={x['nu']:g}: {x['test_f1_macro']:.3f}/{x['sparsity_ratio']:.2f}" for x in r), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
