#!/usr/bin/env python3
"""FT-CUR na Ablação A (N=2000): de onde vêm as sementes que falham? (2026-10-13)

No re-tuning simétrico, o FT-CUR é bimodal: em ~1/5 das sementes o F1 de teste fica em ~0,5-0,65
apesar de F1 de CV 0,93-0,99 com os MESMOS hiperparâmetros. Logo a falha está no ajuste final
(refit no treino inteiro), não na seleção. O refit tem três fontes de aleatoriedade, todas hoje
ligadas a random_state:
  L  sorteio dos landmarks (seletor 'random'; treino interno e contexto de inferência)
  T  torch: inicialização dos pesos e dropout
  S  split interno treino/validação (early stopping)
Para cada caso (dataset, semente) refaz o refit com os best_params publicados variando UM fator
por vez (os outros dois fixos na semente original); o braço 'base' (L=T=S=semente) checa se a
falha se reproduz nesta máquina (o original rodou em GPU T4).

Uso: python scripts/study_ftcur_seed_factors.py [--reps 8] [--n-jobs 4]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import src.models.ft_transformer_cur_wrapper as W  # noqa: E402
from src.data.loaders import DatasetLoader  # noqa: E402
from src.data.preprocessing import _convert_labels, make_splits  # noqa: E402
from src.experiments.runner import _build_model  # noqa: E402
from src.tuning.grids import GRIDS  # noqa: E402
import run_tier1_gridcv as R  # noqa: E402

VARIANT = "FTTransformerCURRandom"
SRC = Path("results/ablation_a_transformers.json")
FAIL_BELOW = 0.85
CONTROLS = {"TWS_2k": [0, 3], "TWC_2k": [0, 3]}   # sementes que deram certo
PATIENCE = 16
R.BUDGET_OVERRIDE.update(epochs=200, patience=PATIENCE)  # protocolo 200/16


def _fit_once(ds, seed, params, arm, seed_L, seed_T, seed_S, patience=16):
    logging.disable(logging.WARNING)
    torch.set_num_threads(1)
    R.BUDGET_OVERRIDE.update(epochs=200, patience=patience)   # de novo: workers do loky não executam o topo do __main__
    X, y, _ = DatasetLoader.load(ds)
    Xtr, Xte, ytr, yte = make_splits(X, y, test_size=0.30, seed=seed)
    fmt = R._label_format(VARIANT)
    ytr, yte = _convert_labels(ytr, fmt), _convert_labels(yte, fmt)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    cfg = GRIDS[VARIANT]
    est, _ = _build_model(cfg["model_name"], {**R._apply_budget(dict(cfg["fixed"])), **params}, label_format=fmt)
    est.set_params(random_state=seed_S)            # S: governa o split interno

    orig_sel = est._landmark_idx                    # L: seletor com semente própria
    def sel(Xs, m, ys=None):
        rs = est.random_state; est.random_state = seed_L
        try:
            return orig_sel(Xs, m, ys)
        finally:
            est.random_state = rs
    est._landmark_idx = sel

    orig_cls = W.FTTransformerClassifier            # T: torch semeado na construção do modelo
    def cls(*a, **k):
        torch.manual_seed(seed_T)
        return orig_cls(*a, **k)
    W.FTTransformerClassifier = cls
    try:
        t0 = time.perf_counter()
        est.fit(Xtr, ytr)
        dt = time.perf_counter() - t0
    finally:
        W.FTTransformerClassifier = orig_cls
    yp = est.predict(Xte)
    if fmt == "signed":
        yte, yp = (yte + 1) // 2, (yp + 1) // 2
    return dict(dataset=ds, seed=seed, arm=arm, seed_L=seed_L, seed_T=seed_T, seed_S=seed_S,
                test_f1=float(f1_score(yte, yp, average="macro", zero_division=0)),
                best_epoch=getattr(est, "best_epoch_", None), n_epochs=getattr(est, "n_epochs_", None),
                stopped_early=getattr(est, "stopped_early_", None),
                epochs_budget=est.epochs, patience=est.patience, fit_time_s=dt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--base-only", action="store_true", help="só o braço base (sementes originais)")
    ap.add_argument("--patience", type=int, default=16, help="200 = sem parada antecipada (treina as 200 épocas)")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--output", type=Path, default=Path("results/study_ftcur_seed_factors.json"))
    a = ap.parse_args()

    pub = [r for r in json.loads(SRC.read_text()) if r["variant"] == VARIANT and r.get("status") == "ok"]
    cases = [r for r in pub if r["test_f1_macro"] < FAIL_BELOW
             or r["seed"] in CONTROLS.get(r["dataset"], [])]
    jobs = []
    for r in cases:
        s, p, ds = r["seed"], r["best_params"], r["dataset"]
        jobs.append((ds, s, p, "base", s, s, s))
        for k in range(0 if a.base_only else a.reps):
            alt = 1000 + 100 * s + k
            jobs += [(ds, s, p, "vary_L", alt, s, s), (ds, s, p, "vary_T", s, alt, s), (ds, s, p, "vary_S", s, s, alt)]
    print(f"{len(cases)} casos, {len(jobs)} ajustes", flush=True)
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(_fit_once)(*j, patience=a.patience) for j in jobs)
    pubf1 = {(r["dataset"], r["seed"]): r["test_f1_macro"] for r in cases}
    for x in res:
        x["published_f1"] = pubf1[(x["dataset"], x["seed"])]
    a.output.write_text(json.dumps(res, indent=1))
    print(f"-> {a.output}")

    # resumo: fração de ajustes que "falham" (F1 < 0,85) por braço, nos casos que falharam no publicado
    for grp, sel in (("falhas", lambda c: c < FAIL_BELOW), ("controles", lambda c: c >= FAIL_BELOW)):
        print(f"\n== {grp} no publicado ==")
        for arm in ("base", "vary_L", "vary_T", "vary_S"):
            f = [x["test_f1"] for x in res if x["arm"] == arm and sel(x["published_f1"])]
            if f:
                print(f"  {arm:7s} n={len(f):3d}  F1 médio {np.mean(f):.3f}  falham {np.mean(np.array(f) < FAIL_BELOW):.0%}")


if __name__ == "__main__":
    main()
