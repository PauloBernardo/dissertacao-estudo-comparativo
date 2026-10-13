#!/usr/bin/env python3
"""Grade completa dos LSSVMs esparsos no Tier 1: F1 de CV, esparsidade e tempo por combinação.

Pergunta (2026-10-06). A seleção por GridSearchCV maximiza só o F1, e:
  (1) nos modelos l1 a esparsidade é subproduto que o critério penaliza -- os vencedores se
      concentram no menor lambda da grade (DualFISTA no Tier 2: 88%) --, enquanto os modelos
      estruturais (Nyström, PCP) têm piso de esparsidade garantido pela construção;
  (2) os l1 gastariam boa parte do tempo em combinações ruins.
Esta análise registra, para CADA combinação, F1 médio de CV, esparsidade média e tempo médio de
ajuste, o que permite (a) comparar modelos com esparsidade equiparada / fronteira F1 x esparsidade,
(b) testar se as combinações lentas são também as ruins e (c) testar a censura da grade em lambda.

Protocolo: mesmo hold-out 70/30 estratificado (random_state=seed) e mesmo StratifiedKFold(5) do
Tier 1; só o conjunto de treino é usado (nada do teste). Grades = as do Tier 1 publicado, com lambda
ESTENDIDO para baixo (campo ``in_published_grid`` marca o que pertence à grade publicada).

Uso: python scripts/study_l1_selection_grid.py [--seeds 0 1 2 3 4] [--n-jobs 4]
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.loaders import DatasetLoader  # noqa: E402
from src.data.preprocessing import _convert_labels  # noqa: E402
from src.experiments.runner import _build_model  # noqa: E402
from src.tuning.grids import GRIDS  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tier2_gridcv import _subsample  # noqa: E402  (mesmo protocolo do Tier 2)

DATASETS = ["BCW", "PID", "HAB", "VCP", "GCR", "AUS", "AI4I", "TWS", "TWM", "TWC"]
# estendido para baixo (censura) e no intervalo entre 0,1 e 1 (fronteira F1 x esparsidade)
LAMBDA_EXT = [0.5, 0.3, 0.2, 0.05, 1e-3, 1e-4]

# grade publicada no Tier 1 (ver run_ablation_a_retune.py para as divergências com grids.py)
PUBLISHED = {
    "DualFISTA":          GRIDS["DualFISTA"]["grid"],
    "FISTANesterov":      GRIDS["FISTANesterov"]["grid"],
    "ADMMNesterovLSSVM":  {**GRIDS["ADMMNesterovLSSVM"]["grid"], "lambda_": [1.0, 0.1, 0.01]},
    "ADMMNystromRandom":  GRIDS["ADMMNystromRandom"]["grid"],
    "FISTANystromRandom": GRIDS["FISTANystromRandom"]["grid"],
    "PCPLSSVm":           GRIDS["PCPLSSVm"]["grid"],
    # Nyström-SVM: grade de 96 do publicado (run_nystrom_random_ablation.GRID_96)
    "NystromLSSVMRandom": {"sigma": [0.1, 0.5, 2.0, 8.0], "gamma": [0.1, 1.0, 10.0, 30.0, 50.0, 100.0],
                           "m_ratio": [0.05, 0.10, 0.20, 0.30]},
}
# formato de rótulo de cada variante no Tier 1 publicado
LABEL_FORMAT = {"ADMMNystromRandom": "binary", "FISTANystromRandom": "binary"}
L1_MODELS = {"DualFISTA", "FISTANesterov", "ADMMNesterovLSSVM", "ADMMNystromRandom", "FISTANystromRandom"}


def _grid(variant: str) -> tuple[dict, set]:
    pub = {k: list(v) for k, v in PUBLISHED[variant].items()}
    ext = {k: list(v) for k, v in pub.items()}
    if variant in L1_MODELS:
        ext["lambda_"] = sorted(set(ext["lambda_"]) | set(LAMBDA_EXT), reverse=True)
    pub_keys = {tuple(sorted(zip(pub, c))) for c in itertools.product(*pub.values())}
    return ext, pub_keys


def _eval_combo(variant, params, X, y, seed, fmt):
    logging.disable(logging.WARNING)
    cfg = GRIDS[variant]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    f1s, sps, ts, its, convs = [], [], [], [], []
    for tr, va in cv.split(X, y):
        sc = StandardScaler().fit(X[tr])
        est, _ = _build_model(cfg["model_name"], {**cfg["fixed"], **params}, label_format=fmt)
        if "random_state" in est.get_params():
            est.set_params(random_state=seed)
        t0 = time.perf_counter()
        est.fit(sc.transform(X[tr]), y[tr])
        ts.append(time.perf_counter() - t0)
        yp = est.predict(sc.transform(X[va]))
        yt = y[va]
        if fmt == "signed":
            yt, yp = (yt + 1) // 2, (yp + 1) // 2
        f1s.append(f1_score(yt, yp, average="macro", zero_division=0))
        sps.append(float(getattr(est, "sparsity_ratio_", np.nan)))
        its.append(getattr(est, "n_iter_", None))
        convs.append(getattr(est, "converged_", None))
    return dict(params=params, cv_f1=float(np.mean(f1s)), sparsity=float(np.mean(sps)),
                fit_time_s=float(np.mean(ts)), n_iter=its, converged=convs)


def main() -> None:
    ap = argparse.ArgumentParser()
    # Só os l1: a esparsidade do PCP e do Nyström-SVM é fixada pela construção (posto, m_ratio) e o
    # ponto deles já está no Tier 1 publicado; ficam em PUBLISHED apenas para uso explícito via --models.
    ap.add_argument("--models", nargs="+", default=sorted(L1_MODELS))
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=None,
                    help="subamostra como no Tier 2 (teto total = n_train/0,7, estratificado por semente)")
    ap.add_argument("--output", type=Path, default=Path("results/study_l1_grid_tier1.json"))
    a = ap.parse_args()

    out = json.loads(a.output.read_text()) if a.output.exists() else []
    done = {(r["variant"], r["dataset"], r["seed"]) for r in out}
    for variant in a.models:
        grid, pub_keys = _grid(variant)
        combos = [dict(zip(grid, c)) for c in itertools.product(*grid.values())]
        fmt = LABEL_FORMAT.get(variant, "signed")
        for ds in a.datasets:
            X, yraw, _ = DatasetLoader.load(ds)
            y = _convert_labels(yraw, fmt)
            for seed in a.seeds:
                if (variant, ds, seed) in done:
                    continue
                Xs, ys = (X, y) if a.n_train is None else _subsample(X, y, round(a.n_train / 0.70), seed)
                Xtr, _, ytr, _ = train_test_split(Xs, ys, test_size=0.30, stratify=ys, random_state=seed)
                t0 = time.time()
                res = Parallel(n_jobs=a.n_jobs)(delayed(_eval_combo)(variant, p, Xtr, ytr, seed, fmt) for p in combos)
                for r in res:
                    r["in_published_grid"] = tuple(sorted(r["params"].items())) in pub_keys
                out.append(dict(variant=variant, dataset=ds, seed=seed, n_train=len(ytr), grid=res))
                a.output.write_text(json.dumps(out))
                print(f"{variant:20s} {ds:5s} seed{seed}  {len(combos)} combos  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
