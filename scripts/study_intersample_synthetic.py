#!/usr/bin/env python3
"""Sintético onde o CONTEXTO (outras amostras, sem rótulo) é necessário — a atenção inter-instâncias
do SAINT (MISA) e do FT-CUR deveria ajudar aqui, se é capaz de usar contexto.

Os Transformers só veem X_train (sem rótulos) como contexto na inferência, então a tarefa é
"relativa ao grupo": G grupos com deslocamento μ_g ~ N(0, s²I), x = μ_g + z, z ~ N(0, I) e
y = 1[z₁ > 0]. Sem contexto, x₁ = μ_g1 + z₁ é ambíguo; com as amostras do mesmo grupo no lote
dá para estimar μ_g e subtrair. Controle: y = 1[x₁ > 0] (contexto irrelevante).
"grupo_novo": os grupos do TESTE são novos — só resolve quem usa o contexto do lote de inferência.
Oráculo: subtrai a média dos k vizinhos de treino mais próximos e limiariza a 1ª coordenada.

    .venv/bin/python scripts/study_intersample_synthetic.py --n-train 50
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
import src.models.ft_transformer_model as M  # noqa: E402
import src.models.ft_transformer_cur_wrapper as W  # noqa: E402
from src.models.ft_transformer_saint_wrapper import SAINTColnorm  # noqa: E402
from src.models.ft_transformer_cur_wrapper import FTTransformerCURColnorm  # noqa: E402
from src.models.transformers.ft_transformer import FTTransformer  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

ORIG_SAINT_FWD = M.SAINTStage.forward
ORIG_FTC = W.FTTransformerClassifier


def fwd_no_misa(self, x):           # style="reference" sem r = r + MISA(LN(r))
    b, n, d = x.shape
    x = x + self.msa(self.ln1(x)); x = x + self.ff1(self.ln2(x))
    r = x.reshape(1, b, n * d); r = r + self.ff2(self.ln4(r))
    return r.reshape(b, n, d)


def make_data(task, n_tr, n_te, d, G, spread, rng):
    mu = rng.normal(0, spread, (G, d))
    def draw(n):
        nonlocal mu
        if task == "grupo_novo" and n == n_te:     # teste: G grupos NOVOS (deslocamentos nunca vistos)
            mu = rng.normal(0, spread, (G, d))
        g = rng.integers(0, G, n); z = rng.standard_normal((n, d)); x = mu[g] + z
        y = (z[:, 0] > 0) if task.startswith("grupo") else (x[:, 0] > 0)
        return x.astype(np.float64), y.astype(int)
    while True:                      # garante as duas classes no treino
        Xtr, ytr = draw(n_tr)
        if 0 < ytr.sum() < n_tr:
            break
    Xte, yte = draw(n_te)
    return Xtr, ytr, Xte, yte


def oracle(Xtr, Xte, k, pool_test=False):
    """pool_test: vizinhos procurados no próprio teste (sem a amostra), p/ grupos novos."""
    P = Xte if pool_test else Xtr
    nn = NearestNeighbors(n_neighbors=min(k + pool_test, len(P))).fit(P)
    _, idx = nn.kneighbors(Xte)
    idx = idx[:, 1:] if pool_test else idx
    return ((Xte[:, 0] - P[idx, 0].mean(1)) > 0).astype(int)


def models(seed, n_tr):
    ep = dict(epochs=200, patience=200)
    def saint(misa):
        M.SAINTStage.forward = ORIG_SAINT_FWD if misa else fwd_no_misa
        return SAINTColnorm(random_state=seed, batch_size=256, **ep)
    def ftcur(inter):
        W.FTTransformerClassifier = ORIG_FTC if inter else (lambda **kw: ORIG_FTC(**{**kw, "use_inter_instance": False}))
        return FTTransformerCURColnorm(random_state=seed, m_ratio=1.0, selection_method="random", **ep)
    return {
        "SAINT (com MISA)": lambda: saint(True),
        "SAINT (sem MISA)": lambda: saint(False),
        "FT-CUR (com inter, m=N)": lambda: ftcur(True),
        "FT-CUR (sem inter)": lambda: ftcur(False),
        "FT-Softmax": lambda: FTTransformer(random_state=seed, attention_type="softmax", batch_size=256,
                                            max_epochs=200, patience=200),
        "XGBoost": lambda: XGBClassifier(n_estimators=200, max_depth=3, random_state=seed, verbosity=0),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=50)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--d", type=int, default=4)
    p.add_argument("--groups", type=int, default=10)
    p.add_argument("--spread", type=float, default=3.0)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--tasks", nargs="+", default=["grupo", "controle"])
    p.add_argument("--output", default=str(ROOT / "results/study_intersample_synthetic.json"))
    a = p.parse_args()
    out = Path(a.output); recs = []
    for task in a.tasks:
        for seed in a.seeds:
            rng = np.random.default_rng(seed)
            Xtr, ytr, Xte, yte = make_data(task, a.n_train, a.n_test, a.d, a.groups, a.spread, rng)
            row = {"oráculo (k-NN)": f1_score(yte, oracle(Xtr, Xte, max(2, a.n_train // a.groups), task == "grupo_novo"), average="macro")}
            for name, mk in models(seed, a.n_train).items():
                t0 = time.perf_counter(); torch.manual_seed(seed)
                clf = mk().fit(Xtr, ytr)
                row[name] = float(f1_score(yte, clf.predict(Xte), average="macro"))
                recs.append({"task": task, "seed": seed, "model": name, "f1": row[name],
                             "n_train": a.n_train, "time_s": time.perf_counter() - t0})
            recs.append({"task": task, "seed": seed, "model": "oráculo (k-NN)", "f1": row["oráculo (k-NN)"],
                         "n_train": a.n_train})
            M.SAINTStage.forward = ORIG_SAINT_FWD; W.FTTransformerClassifier = ORIG_FTC
            print(task, seed, "  ".join(f"{k} {v:.3f}" for k, v in row.items()), flush=True)
            out.write_text(json.dumps(recs, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
