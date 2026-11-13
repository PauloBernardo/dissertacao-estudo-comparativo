#!/usr/bin/env python3
"""Teste de troca de contexto: a predição de um modelo JÁ TREINADO depende do contexto?

As ablações anteriores retreinavam a rede sem a atenção entre amostras (mostra que ela não
AJUDA). Aqui o modelo é treinado uma vez e só o contexto da predição é trocado — mede se ela
é USADA. Os dois modelos predizem sobre [X_ctx ‖ X_teste] (FT-CUR em contexto completo, como nos
Cenários 1 e 2; SAINT em lotes, como no artigo). Contextos (mesmo tamanho do treino):
  original   — o treino (o que o modelo viu)
  bootstrap  — reamostragem com reposição do treino (mesma distribuição, linhas diferentes)
  colperm    — cada coluna permutada independentemente (marginais iguais, estrutura conjunta destruída)
  ruido      — N(0, 1) (dados padronizados)
Métricas contra o original: |Δlogit| médio relativo ao desvio dos logits, fração de classes
trocadas e ΔF1-macro.

    .venv/bin/python scripts/study_context_swap.py --seeds 0 1 2
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT)]
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from src.data.loaders import DatasetLoader  # noqa: E402
from src.data.preprocessing import make_splits  # noqa: E402
from src.models.ft_transformer_cur_wrapper import FTTransformerCURColnorm  # noqa: E402
from src.models.ft_transformer_saint_wrapper import SAINTColnorm  # noqa: E402

# Fixos das grades do Cenário 1 (src/tuning/grids.py); o restante no padrão do wrapper.
FIX = dict(d_model=32, lr=1e-3, epochs=40, patience=6, early_stop_metric="val_loss")
MODELS = {
    "FT-CUR": lambda s: FTTransformerCURColnorm(random_state=s, batch_size=4096, m_ratio=0.2,
                                                selection_method="random", **FIX),
    "SAINT":  lambda s: SAINTColnorm(random_state=s, batch_size=1024, **FIX),
}


def contexts(Xtr, rng):
    n, p = Xtr.shape
    return {
        "bootstrap": Xtr[rng.integers(0, n, n)],
        "colperm":   np.column_stack([Xtr[rng.permutation(n), j] for j in range(p)]),
        "ruido":     rng.standard_normal((n, p)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["BCW", "PID", "HAB", "VCP", "GCR", "AUS", "TWS", "TWM", "TWC"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--output", default=str(ROOT / "results/study_context_swap.json"))
    a = ap.parse_args()
    out = Path(a.output)
    recs = json.loads(out.read_text()) if out.exists() else []
    done = {(r["ds"], r["seed"], r["model"]) for r in recs}
    for ds in a.datasets:
        X, y, _ = DatasetLoader.load(ds); y = np.asarray(y).astype(int)
        for seed in a.seeds:
            Xtr, Xte, ytr, yte = make_splits(X, y, test_size=0.30, seed=seed)
            sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            for name, build in MODELS.items():
                if (ds, seed, name) in done:
                    continue
                torch.manual_seed(seed); np.random.seed(seed); t0 = time.perf_counter()
                clf = build(seed).fit(Xtr, ytr)
                base = clf._logits(Xte); pb = (base > 0).astype(int)
                f1b = f1_score(yte, pb, average="macro"); sd = float(np.std(base)) or 1.0
                rec = {"ds": ds, "seed": seed, "model": name, "f1_original": float(f1b),
                       "n_train": len(ytr), "fit_s": time.perf_counter() - t0}
                orig = clf.X_train_
                for cname, Xc in contexts(Xtr, np.random.default_rng(1000 + seed)).items():
                    clf.X_train_ = Xc
                    lg = clf._logits(Xte); pc = (lg > 0).astype(int)
                    rec[cname] = {"dlogit_rel": float(np.mean(np.abs(lg - base)) / sd),
                                  "flip": float(np.mean(pc != pb)),
                                  "df1": float(f1_score(yte, pc, average="macro") - f1b)}
                clf.X_train_ = orig
                recs.append(rec); out.write_text(json.dumps(recs, indent=1))
                print(f"{ds:4s} {seed} {name:6s} F1 {f1b:.3f}  " + "  ".join(
                    f"{c}: Δlog {rec[c]['dlogit_rel']:.3f} flip {rec[c]['flip']:.3f} ΔF1 {rec[c]['df1']:+.3f}"
                    for c in ("bootstrap", "colperm", "ruido")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
