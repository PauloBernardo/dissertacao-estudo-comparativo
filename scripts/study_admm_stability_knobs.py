#!/usr/bin/env python3
"""Os três controles de estabilidade do ADMM: λ₂ (Elastic Net), η (restart) e teto.

Contexto
--------
O artigo-fonte reporta "convergence stability problems" e sugere, como trabalho
futuro, uma regra de restart. A implementação desta dissertação tem DOIS
remédios para isso — o termo ℓ2 do Elastic Net (que injeta convexidade forte e
satisfaz a condição de Goldstein) e o restart adaptativo de Lyapunov — mas
nenhum dos dois foi calibrado: λ₂ está fixo em 1e-3 e η em 0,999. Como η→1 faz
a condição c_k < η·c_{k-1} ser quase sempre satisfeita, o restart praticamente
nunca dispara.

Este estudo mede o efeito de cada controle na TAXA DE CONVERGÊNCIA — que é o
que eles prometem — em vez de em F1, que não prometem. Desenho
um-fator-a-cada-vez a partir de uma linha de base, no ponto de operação onde a
não-convergência é mais sensível (λ=0,1: ~47% das execuções atingem o teto).

Saída: results/admm_stability_knobs.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loaders import DatasetLoader                       # noqa: E402
from src.data.preprocessing import _convert_labels, make_splits  # noqa: E402
from src.models.lssvm.primal.admm_nesterov import ADMMNesterovLSSVM  # noqa: E402

# (σ, τ) modais do GridSearchCV; λ=0,1 é o ponto de máxima sensibilidade.
PARAMS = {"CREDIT": (2.0, 5.0), "HIGGS50K": (2.0, 0.5)}
LAMBDA = 0.1
N_TOTAL_CAP = 2857          # protocolo do Tier 2 (N_treino = 2000)
CAP = 3000                  # teto folgado para observar a convergência
CAP_TIGHT = 500             # teto atual da dissertação

# Braços: (rótulo, lambda2_, restart_eta, max_iter)
ARMS = [
    ("base            ", 0.0,   0.999, CAP),        # sem ℓ2, restart quase inativo
    ("teto 500        ", 0.0,   0.999, CAP_TIGHT),  # quanto o teto atual censura
    ("l2=1e-3 (atual) ", 1e-3,  0.999, CAP),
    ("l2=1e-2         ", 1e-2,  0.999, CAP),
    ("l2=1e-1         ", 1e-1,  0.999, CAP),
    ("eta=0.9         ", 0.0,   0.9,   CAP),
    ("eta=0.5 (Marinho)", 0.0,  0.5,   CAP),
]


def prepare(dataset: str, seed: int):
    X, y, _ = DatasetLoader.load(dataset)
    n = min(N_TOTAL_CAP, len(y))
    idx, _ = next(StratifiedShuffleSplit(1, train_size=n, random_state=seed).split(X, y))
    X, y = X[idx], y[idx]
    Xtr, Xte, ytr, yte = make_splits(X, y, test_size=0.30, seed=seed)
    sc = StandardScaler().fit(Xtr)
    return (sc.transform(Xtr), sc.transform(Xte),
            _convert_labels(ytr, "signed"), _convert_labels(yte, "signed"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--datasets", nargs="+", default=list(PARAMS))
    p.add_argument("--output", type=Path,
                   default=ROOT / "results" / "admm_stability_knobs.json")
    a = p.parse_args()
    logging.getLogger().setLevel(logging.ERROR)

    recs = json.loads(a.output.read_text()) if a.output.exists() else []
    done = {(r["dataset"], r["seed"], r["arm"]) for r in recs}

    for ds in a.datasets:
        sigma, tau = PARAMS[ds]
        for seed in a.seeds:
            Xtr, Xte, ytr, yte = prepare(ds, seed)
            for label, l2, eta, cap in ARMS:
                arm = label.strip()
                if (ds, seed, arm) in done:
                    continue
                m = ADMMNesterovLSSVM(sigma=sigma, tau=tau, lambda_=LAMBDA,
                                      lambda2_=l2, restart_eta=eta,
                                      max_iter=cap, rho=None)
                t0 = time.perf_counter()
                m.fit(Xtr, ytr)
                dt = time.perf_counter() - t0
                rec = {"dataset": ds, "seed": seed, "arm": arm,
                       "lambda2": l2, "eta": eta, "max_iter": cap,
                       "n_iter": int(m.n_iter_), "converged": bool(m.converged_),
                       "sparsity": float(m.sparsity_ratio_),
                       "f1_macro": float(f1_score(yte, m.predict(Xte), average="macro")),
                       "fit_time_s": dt}
                recs.append(rec)
                a.output.write_text(json.dumps(recs, indent=1), encoding="utf-8")
                print(f"{ds:9s} s{seed} {label} | {rec['n_iter']:5d}it "
                      f"{'ok ' if rec['converged'] else 'NAO'} "
                      f"esp={rec['sparsity']:.3f} F1={rec['f1_macro']:.4f} "
                      f"{dt:5.1f}s", flush=True)
    print(f"\n-> {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
