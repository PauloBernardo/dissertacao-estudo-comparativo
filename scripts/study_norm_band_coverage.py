#!/usr/bin/env python3
"""Por que seleção por faixa de norma falha em variedade de baixa dimensão.

Seleção por ‖x_i‖² escolhe uma CASCA: nos sintéticos 2D centrados, ‖x‖ é o raio, logo
uma faixa de norma é um anel. Anel não cobre espiral nem tabuleiro. Isso unifica duas
falhas que pareciam distintas: o `colnorm` pega o anel externo, o `midband` o anel do
meio (mais estreito, pior cobertura). Só o k-means — e o `opposite`, que é k-means em
feature space — espalha os protótipos pela variedade.

Mede, por seletor: (a) erro de quantização, a grandeza que governa o erro de Nyström
segundo Zhang, Tsang & Kwok (2008); (b) a extensão radial coberta pelos landmarks.

Uso:
    python scripts/study_norm_band_coverage.py [--m-ratio 0.10] [--seeds 5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loaders import DatasetLoader            # noqa: E402
from src.models.landmark_selection import get_selector  # noqa: E402

METODOS = ["kmeans", "random", "colnorm", "midband", "colnorm_inv"]
GEO = ["TWS", "TWM", "TWC"]
TABULARES = ["BCW", "PID", "HAB", "AUS"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m-ratio", type=float, default=0.10)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--datasets", nargs="+", default=GEO + TABULARES)
    ap.add_argument("--output", type=Path, default=Path("results/norm_band_coverage.json"))
    args = ap.parse_args()

    out = []
    for ds in args.datasets:
        X, _, _ = DatasetLoader.load(ds)
        X = StandardScaler().fit_transform(X)
        m = max(2, int(args.m_ratio * len(X)))
        raio = np.sqrt((X ** 2).sum(1))
        for met in METODOS:
            qe, r_lo, r_hi = [], [], []
            for seed in range(args.seeds):
                sel = get_selector(met, m, seed)
                sel.fit(X)
                L = X[sel.indices_]
                # erro de quantizacao: distancia media ao landmark mais proximo
                d = np.sqrt(((X[:, None, :] - L[None, :, :]) ** 2).sum(-1)).min(1)
                qe.append(float(d.mean()))
                rl = raio[sel.indices_]
                r_lo.append(float(rl.min())); r_hi.append(float(rl.max()))
            rec = dict(dataset=ds, tipo="geometrico" if ds in GEO else "tabular",
                       metodo=met, n=int(len(X)), m=m, m_ratio=args.m_ratio,
                       erro_quantizacao=round(float(np.mean(qe)), 4),
                       raio_landmarks=[round(float(np.mean(r_lo)), 3),
                                       round(float(np.mean(r_hi)), 3)],
                       raio_dados=[round(float(raio.min()), 3), round(float(raio.max()), 3)])
            rec["cobertura_radial"] = round(
                (rec["raio_landmarks"][1] - rec["raio_landmarks"][0])
                / max(rec["raio_dados"][1] - rec["raio_dados"][0], 1e-12), 3)
            out.append(rec)
            print(json.dumps(rec), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))
    print(f"\nescrito {args.output} ({len(out)} registros)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
