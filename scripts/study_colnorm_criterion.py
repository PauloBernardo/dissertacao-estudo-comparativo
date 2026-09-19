#!/usr/bin/env python3
"""O que o seletor `colnorm` realmente mede.

O seletor amostra com probabilidade proporcional a ‖x_i‖² no espaço de ENTRADA
(`landmark_selection.ColumnNormSelector` com kernel=None; ver também a decisão
explícita em `nystrom.py`, onde só `leverage` recebe o kernel, para não pagar a
matriz N×N). O nome e a citação (Drineas) referem-se à amostragem por norma de
COLUNA da matriz-alvo. Este script mede, nos dados e nos sigmas do estudo, o
quanto ‖x_i‖² se relaciona com:

  (a) ‖K[:,i]‖², a norma de coluna verdadeira do kernel RBF — o que o nome promete;
  (b) o leverage score de posto m de K — o "padrão-ouro" que o Apêndice lamenta
      não ter comparado.

Uso:
    python scripts/study_colnorm_criterion.py [--n-max 800] [--m-ratio 0.2]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loaders import DatasetLoader  # noqa: E402

# Grade de sigma do estudo (LSSVMs) mais os valores que o GridCV mais seleciona.
SIGMAS = [0.1, 0.5, 2.0, 5.0, 8.0]
DATASETS = ["BCW", "PID", "HAB", "VCP", "AUS", "GCR", "AI4I", "TWS", "TWM", "TWC"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-max", type=int, default=800,
                    help="Subamostra datasets maiores (a matriz K é N×N).")
    ap.add_argument("--m-ratio", type=float, default=0.20,
                    help="Posto m/n para o leverage score.")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--output", type=Path, default=Path("results/colnorm_criterion.json"))
    args = ap.parse_args()

    out = []
    for ds in args.datasets:
        X, _, _ = DatasetLoader.load(ds)
        X = StandardScaler().fit_transform(X)
        if len(X) > args.n_max:
            rng = np.random.default_rng(0)
            X = X[rng.choice(len(X), args.n_max, replace=False)]
        # ‖x_i‖²: exatamente o que ColumnNormSelector usa como probabilidade.
        x_norm = (X ** 2).sum(1)
        D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        m = max(2, int(args.m_ratio * len(X)))
        for sigma in SIGMAS:
            K = np.exp(-D / (2 * sigma ** 2))
            col_norm = (K ** 2).sum(0)              # ‖K[:,i]‖², o critério de Drineas
            w, V = np.linalg.eigh(K)                # K é simétrica PSD
            leverage = (V[:, -m:] ** 2).sum(1)      # leverage de posto m
            rec = dict(
                dataset=ds, n=int(len(X)), sigma=sigma, m=m,
                rho_colnorm=_rho(x_norm, col_norm),
                rho_leverage=_rho(x_norm, leverage),
            )
            out.append(rec)
            print(json.dumps(rec), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))
    print(f"\nescrito {args.output} ({len(out)} registros)")
    return 0


def _rho(a: np.ndarray, b: np.ndarray) -> float | None:
    """Spearman, com None quando um dos vetores é constante (K ≈ identidade)."""
    r = spearmanr(a, b).statistic
    return None if not np.isfinite(r) else round(float(r), 4)


if __name__ == "__main__":
    raise SystemExit(main())
