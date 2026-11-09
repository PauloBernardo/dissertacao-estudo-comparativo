#!/usr/bin/env python3
"""Ablação da atenção entre instâncias em regime de ESCASSEZ (sintéticos 2D do Cenário 1).

Hipótese: com poucas amostras de treino, olhar as demais amostras (atenção entre instâncias)
deveria importar mais. Para TWS/TWM/TWC: split estratificado 70/30 do Cenário 1 (mesma semente),
treino subamostrado (estratificado) para --n-train amostras; teste = os 30% (120 amostras).
Modelos (mesmos construtores de study_intersample_synthetic.py; 200 épocas, sem parada antecipada):
SAINT com/sem MISA, FT-CUR com/sem atenção inter-instâncias (m = N: todos os pontos de treino
como landmarks), FT-Softmax e XGBoost como referências sem contexto.

    .venv/bin/python scripts/study_intersample_scarce.py --n-train 100
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from src.data.loaders import DatasetLoader  # noqa: E402
from src.data.preprocessing import make_splits  # noqa: E402
import study_intersample_synthetic as S  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=100)
    p.add_argument("--datasets", nargs="+", default=["TWS", "TWM", "TWC"])
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--output", default=None)
    a = p.parse_args()
    out = Path(a.output or ROOT / f"results/study_intersample_scarce_n{a.n_train}.json")
    recs = json.loads(out.read_text()) if out.exists() else []
    done = {(r["ds"], r["seed"], r["model"]) for r in recs}
    for ds in a.datasets:
        X, y, _ = DatasetLoader.load(ds)
        y = np.asarray(y).astype(int)
        for seed in a.seeds:
            Xtr, Xte, ytr, yte = make_splits(X, y, test_size=0.30, seed=seed)
            if a.n_train < len(ytr):
                Xtr, _, ytr, _ = train_test_split(Xtr, ytr, train_size=a.n_train, stratify=ytr,
                                                  random_state=seed)
            sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
            row = {}
            for name, mk in S.models(seed, a.n_train).items():
                if (ds, seed, name) in done:
                    continue
                t0 = time.perf_counter(); torch.manual_seed(seed); np.random.seed(seed)
                clf = mk().fit(Xtr_s, ytr)
                f1 = float(f1_score(yte, clf.predict(Xte_s), average="macro"))
                S.M.SAINTStage.forward = S.ORIG_SAINT_FWD; S.W.FTTransformerClassifier = S.ORIG_FTC
                row[name] = f1
                recs.append({"ds": ds, "seed": seed, "model": name, "f1": f1, "n_train": len(ytr),
                             "time_s": time.perf_counter() - t0})
                out.write_text(json.dumps(recs, indent=1))
            print(ds, seed, "  ".join(f"{k} {v:.3f}" for k, v in row.items()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
