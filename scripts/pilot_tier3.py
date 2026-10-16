"""Piloto do Tier 3 (v2): custo real de um fit por modelo em N = 200 mil.

Uma base, uma semente, hiperparâmetros no meio da grade. Mede tempo de fit,
pico de RAM, tempo de predição e F1 para calibrar a estimativa de custo do
PLANO_FECHAMENTO e escolher m (2000 vs 5000). PCP com r = m (decisão de
2026-10-16: mesmo orçamento de pontos que os Nyström).

Cada (modelo, m) roda em processo próprio — o pico de RAM é medido isolado e
um OOM não derruba o resto. Uso:

    .venv/bin/python scripts/pilot_tier3.py --variant NystromLSSVMRandom --m 2000 \
        --dataset HIGGS500K --seed 0 --output results/pilot_tier3.json
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_tier2_gridcv as g  # noqa: E402

N_TOTAL = 200_000
TEST_SIZE = 0.20  # 160 mil de treino

# Meio da grade de cada modelo (o piloto mede custo, não seleciona).
PARAMS = {
    "NystromLSSVMRandom": {"sigma": 5.0, "gamma": 10.0},
    "ADMMNystromRandom":  {"sigma": 5.0, "tau": 0.1, "lambda_rel": 0.01},
    "FISTANystromRandom": {"sigma": 5.0, "tau": 0.1, "lambda_rel": 0.01},
    "PCPLSSVm":           {"sigma": 5.0, "tau": 1e-3},
    "XGBoost":            {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1},
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=sorted(PARAMS))
    p.add_argument("--m", type=int, default=0, help="landmarks (r no PCP); 0 p/ XGBoost")
    p.add_argument("--dataset", default="HIGGS500K")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    g.set_global_seed(a.seed)
    X, y, _ = g.DatasetLoader.load(a.dataset)
    X, y = g._subsample(X, y, N_TOTAL, a.seed)
    X_tr, X_te, y_tr, y_te = g.make_splits(X, y, test_size=TEST_SIZE, seed=a.seed)
    fmt = g._label_format(a.variant)
    y_tr, y_te = g._convert_labels(y_tr, fmt), g._convert_labels(y_te, fmt)

    pipe, _ = g._build_pipeline(a.variant, a.seed)
    params = dict(PARAMS[a.variant])
    if a.variant == "PCPLSSVm":
        params["rank"] = a.m
    elif a.m:
        params["m_ratio"] = a.m / len(X_tr)
    pipe.set_params(**{f"clf__{k}": v for k, v in params.items()})

    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0
    rss_fit = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    t0 = time.perf_counter()
    metrics = g._compute_test_metrics(pipe, X_te, y_te, fmt)
    pred_s = time.perf_counter() - t0

    rec = {
        "variant": a.variant, "dataset": a.dataset, "seed": a.seed, "m": a.m,
        "n_train": len(X_tr), "n_test": len(X_te), "params": params,
        "fit_s": round(fit_s, 2), "pred_s": round(pred_s, 3),
        "peak_rss_mb": round(rss_fit), "rss_before_fit_mb": round(rss0),
        **metrics, **g._collect_sparsity(pipe.named_steps["clf"], X_te),
    }
    out = Path(a.output)
    recs = json.loads(out.read_text()) if out.exists() else []
    recs.append(rec)
    out.write_text(json.dumps(recs, indent=1))
    print(json.dumps({k: rec[k] for k in ("variant", "m", "fit_s", "peak_rss_mb", "test_f1_macro")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
