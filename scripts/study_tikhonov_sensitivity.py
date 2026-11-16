#!/usr/bin/env python3
"""Sensibilidade à constante de Tikhonov ν dos primais ℓ1 densos (pergunta da banca).

ADMM-Nesterov e FISTA-Nesterov fatoram G̃ = G + νI (Cholesky); no estudo, ν = 0,01 fixo.
Aqui, no Cenário 1 (10 bases, mesma partição 70/30 por semente), cada (base, semente) usa os
hiperparâmetros escolhidos pelo GridSearchCV publicado (results/tier1_gridcv.json) e só ν varia.
Mede F1-macro, esparsidade e o número de condição de G̃ (espectro do kernel de treino).

    .venv/bin/python scripts/study_tikhonov_sensitivity.py --seeds 0 1 2 3 4 5 6 7 8 9
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from src.data.loaders import DatasetLoader  # noqa: E402
from src.data.preprocessing import _convert_labels, make_splits  # noqa: E402
from src.experiments.runner import _build_model  # noqa: E402
from src.tuning.grids import GRIDS  # noqa: E402
from run_tier1_gridcv import _compute_test_metrics  # noqa: E402

DATASETS = ["AI4I", "AUS", "BCW", "GCR", "HAB", "PID", "TWC", "TWM", "TWS", "VCP"]
VARIANTS = ["ADMMNesterovLSSVM", "FISTANesterov"]
NUS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--output", default=str(ROOT / "results/study_tikhonov_sensitivity.json"))
    a = ap.parse_args()
    pub = {(r["variant"], r["dataset"], r["seed"]): r
           for r in json.loads((ROOT / "results/tier1_gridcv.json").read_text())
           if r["variant"] in VARIANTS and r.get("status") == "ok"}
    recs = []
    for ds in DATASETS:
        X, y, _ = DatasetLoader.load(ds)
        for seed in a.seeds:
            Xtr, Xte, ytr_raw, yte_raw = make_splits(X, y, test_size=0.30, seed=seed)
            ytr, yte = _convert_labels(ytr_raw, "signed"), _convert_labels(yte_raw, "signed")
            Xs = StandardScaler().fit(Xtr).transform(Xtr)
            for var in VARIANTS:
                ref = pub[(var, ds, seed)]
                bp = ref["best_params"]
                # espectro do kernel de treino para o σ escolhido
                d2 = (Xs**2).sum(1)[:, None] + (Xs**2).sum(1)[None] - 2 * Xs @ Xs.T
                G = np.exp(-np.maximum(d2, 0) / (2 * bp["sigma"] ** 2))
                ev = np.linalg.eigvalsh(G)
                for nu in NUS:
                    params = {**GRIDS[var]["fixed"], **bp, "tikhonov_reg": nu}
                    est, _ = _build_model(GRIDS[var]["model_name"], params, label_format="signed")
                    pipe = Pipeline([("scaler", StandardScaler()), ("clf", est)])
                    t0 = time.perf_counter(); pipe.fit(Xtr, ytr); ft = time.perf_counter() - t0
                    m = _compute_test_metrics(pipe, Xte, yte, "signed")
                    rec = {"variant": var, "dataset": ds, "seed": seed, "nu": nu, "best_params": bp,
                           "test_f1_macro": m["test_f1_macro"],
                           "sparsity_ratio": float(getattr(pipe.named_steps["clf"], "sparsity_ratio_", np.nan)),
                           "cond": float((ev.max() + nu) / (max(ev.min(), 0) + nu)),
                           "lambda_min": float(ev.min()), "lambda_max": float(ev.max()),
                           "fit_s": ft, "published_f1": ref["test_f1_macro"],
                           "published_sparsity": ref.get("sparsity_ratio")}
                    recs.append(rec)
                print(f"{ds:4s} {seed} {var:18s} " + "  ".join(
                    f"ν={r['nu']:g}: {r['test_f1_macro']:.3f}/{r['sparsity_ratio']:.2f}"
                    for r in recs[-len(NUS):]) + f"  | publ. {ref['test_f1_macro']:.3f}", flush=True)
            Path(a.output).write_text(json.dumps(recs, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
