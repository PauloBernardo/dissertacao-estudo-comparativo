#!/usr/bin/env python3
"""Amostra do Tier 1: os Transformers que pararam cedo perderiam F1 por causa da paciência? (2026-10-14)

No Tier 1 publicado (200/16), 34-67% das execuções de cada Transformer pararam por paciência com
melhor época < 20. No FT-CUR da Ablação A isso era platô inicial: sem paciência, as sementes que
falhavam foram de ~0,5-0,65 para 0,93-0,99 (results/study_ftcur_pat200_base.json). Aqui se mede se
o mesmo vale no Tier 1 ou se a parada cedo é convergência legítima.

Amostra: por Transformer, uma execução com parada precoce por dataset (10 por modelo, semente
sorteada com rng fixo). Cada caso é reajustado com os best_params publicados, duas vezes:
paciência 16 (controle local: a GPU daqui não reproduz a trajetória do Kaggle) e paciência 200
(sem parada antecipada; restaura o checkpoint de melhor validação). Ganho = pac200 - pac16.

Uso: python scripts/study_tf_patience_sample.py [--n-jobs 4]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data.loaders import DatasetLoader  # noqa: E402
from src.data.preprocessing import _convert_labels, make_splits  # noqa: E402
from src.experiments.runner import _build_model  # noqa: E402
from src.tuning.grids import GRIDS  # noqa: E402
import run_tier1_gridcv as R  # noqa: E402

TF = ["FTTransformer_softmax", "FTTransformer_topk", "FTTransformer_entmax", "FTTransformer_sparsemax",
      "SAINTColnorm", "FTTransformerCURRandom"]
EARLY = 20


def _fit(variant, ds, seed, params, patience):
    logging.disable(logging.WARNING)
    torch.set_num_threads(1)
    R.BUDGET_OVERRIDE.clear(); R.BUDGET_OVERRIDE.update(epochs=200, patience=patience)
    X, y, _ = DatasetLoader.load(ds)
    Xtr, Xte, ytr, yte = make_splits(X, y, test_size=0.30, seed=seed)
    fmt = R._label_format(variant)
    ytr, yte = _convert_labels(ytr, fmt), _convert_labels(yte, fmt)
    sc = StandardScaler().fit(Xtr)
    cfg = GRIDS[variant]
    est, _ = _build_model(cfg["model_name"], {**R._apply_budget(dict(cfg["fixed"])), **params}, label_format=fmt)
    if "random_state" in est.get_params():
        est.set_params(random_state=seed)
    t0 = time.perf_counter()
    est.fit(sc.transform(Xtr), ytr)
    dt = time.perf_counter() - t0
    yp = est.predict(sc.transform(Xte))
    if fmt == "signed":
        yte, yp = (yte + 1) // 2, (yp + 1) // 2
    return dict(variant=variant, dataset=ds, seed=seed, patience=patience,
                test_f1=float(f1_score(yte, yp, average="macro", zero_division=0)),
                best_epoch=getattr(est, "best_epoch_", None), n_epochs=getattr(est, "n_epochs_", None),
                fit_time_s=dt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--output", type=Path, default=Path("results/study_tf_patience_sample.json"))
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    pool = defaultdict(list)
    for r in json.load(open("results/tier1_gridcv.json")):
        if (r["variant"] in TF and r.get("status") == "ok" and r.get("stopped_early")
                and (r.get("best_epoch") or 99) < EARLY):
            pool[(r["variant"], r["dataset"])].append(r)
    cases = [lst[rng.integers(len(lst))] for _, lst in sorted(pool.items())]
    print(f"{len(cases)} casos, {2 * len(cases)} ajustes", flush=True)
    jobs = [(c["variant"], c["dataset"], c["seed"], c["best_params"], p) for c in cases for p in (16, 200)]
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(_fit)(*j) for j in jobs)
    pub = {(c["variant"], c["dataset"], c["seed"]): c["test_f1_macro"] for c in cases}
    for x in res:
        x["published_f1"] = pub[(x["variant"], x["dataset"], x["seed"])]
    a.output.write_text(json.dumps(res, indent=1))

    by = {(x["variant"], x["dataset"], x["seed"], x["patience"]): x for x in res}
    print(f"\n{'modelo':24s} n  pub   pac16  pac200  ganho(200-16)  casos com ganho>0,05")
    for v in TF:
        ks = [(vv, d, s) for (vv, d, s, p) in by if vv == v and p == 16]
        if not ks:
            continue
        g = np.array([by[k + (200,)]["test_f1"] - by[k + (16,)]["test_f1"] for k in ks])
        print(f"{v:24s} {len(ks):2d} {np.mean([pub[k] for k in ks]):.3f} "
              f"{np.mean([by[k + (16,)]['test_f1'] for k in ks]):.3f}  {np.mean([by[k + (200,)]['test_f1'] for k in ks]):.3f}"
              f"   {g.mean():+.3f}          {int((g > 0.05).sum())}/{len(g)}")


if __name__ == "__main__":
    main()
