"""Controle do Tier 3: LSSVM denso treinado em 2.000 amostras sorteadas do treino.

Mesmo orçamento de pontos que m = 2000, mas descartando o resto dos 200 mil.
Pergunta: o Nyström ganha F1 por enxergar os 200 mil? Teste = os mesmos 50 mil
do Tier 3. Hiperparâmetros por GridSearchCV 3-fold nas 2 mil (barato), por semente.

    .venv/bin/python scripts/tier3_dense_2k.py --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_tier2_gridcv as g  # noqa: E402
import run_tier3 as t3  # noqa: E402

N_DENSE = 2000
GRID = {"clf__sigma": [0.15, 0.5, 1.5, 5.0, 15.0, 50.0],
        "clf__tau": [0.1, 0.5, 2.5, 12.5, 50.0, 250.0, 1000.0]}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--datasets", nargs="+", default=t3.DATASETS)
    p.add_argument("--output", type=Path, default=ROOT / "results" / "tier3_dense_2k.json")
    a = p.parse_args()

    fmt = g._label_format("StandardLSSVM")
    recs = []
    for d in a.datasets:
        for s in a.seeds:
            g.set_global_seed(s)
            X_tr, X_te, y_tr, y_te, _ = t3.load_split(d, s, fmt)
            X_2k, _, y_2k, _ = train_test_split(X_tr, y_tr, train_size=N_DENSE,
                                                stratify=y_tr, random_state=s)
            pipe, _ = g._build_pipeline("StandardLSSVM", s)
            gs = GridSearchCV(pipe, GRID, cv=3, scoring="f1_macro", n_jobs=4).fit(X_2k, y_2k)
            t0 = time.perf_counter()
            pipe.set_params(**gs.best_params_).fit(X_2k, y_2k)
            fit_s = time.perf_counter() - t0
            f1 = t3._f1(y_te, pipe.predict(X_te), fmt)
            rec = {"dataset": d, "seed": s, "n_train": N_DENSE, "n_test": len(X_te),
                   "params": gs.best_params_, "cv_f1": gs.best_score_,
                   "test_f1_macro": f1, "fit_time_s": fit_s}
            recs.append(rec)
            print(json.dumps(rec), flush=True)
            a.output.write_text(json.dumps(recs, indent=1))
    for d in a.datasets:
        v = [r["test_f1_macro"] for r in recs if r["dataset"] == d]
        print(f"{d}: F1 = {np.mean(v):.4f} ± {np.std(v, ddof=1):.4f} (n={len(v)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
