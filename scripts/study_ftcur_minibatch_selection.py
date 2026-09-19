#!/usr/bin/env python3
"""FT-CUR em mini-lote: o seletor de landmarks importa quando ele é de fato usado?

No caminho de mini-lote histórico (`minibatch_landmarks="per_batch"`) os landmarks
escolhidos no `fit` são DESCARTADOS: dentro de cada lote o sorteio é uniforme
(`torch.randperm(B)[:m]`). Logo o seletor não tem efeito algum ali. O modo "global"
anexa a cada lote os m landmarks escolhidos no treino inteiro, e só então o critério
de seleção passa a valer.

Este estudo compara, com m_ratio FIXO em 10% e mini-lote forçado:

  per_batch      — histórico: landmarks aleatórios dentro do lote (o seletor é ignorado)
  global+random  — controle: landmarks globais, sorteio uniforme
  global+colnorm — o critério em uso: ∝ ‖x_i‖² (periféricos)
  global+colnorm_inv — o invertido: ∝ 1/‖x_i‖², que reproduz a norma de coluna
                   VERDADEIRA do kernel RBF sem montar a matriz N×N
                   (ver scripts/study_colnorm_criterion.py)

Uso:
    python scripts/study_ftcur_minibatch_selection.py --n-train 2000 --batch-size 512
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loaders import DatasetLoader                              # noqa: E402
from src.models.ft_transformer_cur_wrapper import FTTransformerCURColnorm  # noqa: E402

ARMS = {
    "per_batch":          dict(minibatch_landmarks="per_batch", selection_method="colnorm"),
    "global+random":      dict(minibatch_landmarks="global",    selection_method="random"),
    "global+colnorm":     dict(minibatch_landmarks="global",    selection_method="colnorm"),
    "global+colnorm_inv": dict(minibatch_landmarks="global",    selection_method="colnorm_inv"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["BANK", "TELCO", "ADULT", "SHOPPERS"])
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=512,
                    help="Menor que n_fit, para FORÇAR o caminho de mini-lote.")
    ap.add_argument("--m-ratio", type=float, default=0.10)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=16)
    ap.add_argument("--output", type=Path, default=Path("results/ftcur_minibatch_selection.json"))
    args = ap.parse_args()

    out = []
    for ds in args.datasets:
        X, y, _ = DatasetLoader.load(ds)
        for seed in args.seeds:
            Xs, ys = _subsample(X, y, args.n_train, seed)
            X_tr, X_te, y_tr, y_te = train_test_split(
                Xs, ys, test_size=0.30, random_state=seed, stratify=ys)
            sc = StandardScaler().fit(X_tr)
            X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)
            n_fit = round(0.8 * len(X_tr))
            for arm, kw in ARMS.items():
                t0 = time.time()
                try:
                    m = FTTransformerCURColnorm(
                        d_model=32, n_heads=2, n_layers=1, m_ratio=args.m_ratio,
                        lr=1e-3, epochs=args.epochs, patience=args.patience,
                        early_stop_metric="val_loss", batch_size=args.batch_size,
                        random_state=seed, **kw)
                    m.fit(X_tr, y_tr)
                    pred = m.predict(X_te)
                    rec = dict(dataset=ds, seed=seed, arm=arm, n_fit=n_fit,
                               batch_size=args.batch_size, m_ratio=args.m_ratio,
                               minilote=n_fit > args.batch_size,
                               f1_macro=round(f1_score(y_te, pred, average="macro",
                                                       zero_division=0), 4),
                               best_epoch=getattr(m, "best_epoch_", None),
                               n_epochs=getattr(m, "n_epochs_", None),
                               s=round(time.time() - t0, 1), status="ok")
                except Exception as exc:
                    rec = dict(dataset=ds, seed=seed, arm=arm, status="erro",
                               err=f"{type(exc).__name__}: {exc}"[:150],
                               s=round(time.time() - t0, 1))
                out.append(rec)
                print(json.dumps(rec), flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(out, indent=1))
    print(f"\nescrito {args.output} ({len(out)} registros)")
    return 0


def _subsample(X, y, n, seed):
    if n >= len(X):
        return X, y
    Xs, _, ys, _ = train_test_split(X, y, train_size=n, random_state=seed, stratify=y)
    return Xs, ys


if __name__ == "__main__":
    raise SystemExit(main())
