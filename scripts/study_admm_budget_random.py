#!/usr/bin/env python3
"""Orçamento de iterações do ADMM-Nyström com seleção aleatória (2026-10-14).

Pergunta: o teto de 500 iterações trunca o ADMM-Nyström (random) na Ablação D?
Par do estudo de paciência dos Transformers: o critério de parada limita o modelo?

Mesmo runner da Ablação D (run_tier2_fixedparams.run_one_fixed, config random),
N=5000, 30 sementes, variando só max_iter ∈ {500, 5000}. Registra n_iter_ e se o
teto foi atingido. O braço 500 deve reproduzir o canônico célula a célula.
Braços de Elastic Net (λ₂ ∈ {0,01; 0,1}, teto 500) acrescentados no mesmo dia.
Saída: results/admm_budget_random.json (retomável)
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import run_tier2_fixedparams as R  # noqa: E402

import os
OUT = Path(os.environ.get("BUDGET_OUT", ROOT / "results" / "admm_budget_random.json"))
CFG = json.loads((ROOT / "config" / "tier2_fixed_params_random.json").read_text())["ADMMNystromRandom"]
DS = list(CFG)


def main() -> None:
    logging.disable(logging.WARNING)
    seeds = [int(s) for s in sys.argv[1:]] or list(range(30))
    recs = json.loads(OUT.read_text()) if OUT.exists() else []
    done = {(r["dataset"], r["seed"], r["max_iter"], r.get("lambda2", 0.0)) for r in recs}
    orig = R._build_fixed_pipeline
    for mi, l2 in ((500, 0.0), (5000, 0.0), (500, 0.01), (500, 0.1)):
        extra = {"lambda2_": l2} if l2 else {}
        cfg = {"ADMMNystromRandom": {d: {**p, "max_iter": mi, **extra} for d, p in CFG.items()}}
        for ds in DS:
            for s in seeds:
                if (ds, s, mi, l2) in done:
                    continue
                box = {}
                def build(*a, **k):
                    pipe, fp = orig(*a, **k); box["clf"] = pipe.named_steps["clf"]; return pipe, fp
                R._build_fixed_pipeline = build
                r = R.run_one_fixed("ADMMNystromRandom", ds, s, 5000, cfg)
                clf = box.get("clf")
                r.update(max_iter=mi, lambda2=l2, n_iter=int(getattr(clf, "n_iter_", -1)),
                         converged=bool(getattr(clf, "converged_", False)))
                r.pop("traceback", None)
                recs.append(r)
                OUT.write_text(json.dumps(recs, indent=1))
                print(f"{mi:5d} l2={l2:<4} {ds:9s} s{s:<2d} it={r['n_iter']:5d} F1={r.get('test_f1_macro', float('nan')):.4f} "
                      f"esp={r.get('sparsity_ratio', float('nan')):.3f} t={r.get('fit_time_s', 0):.1f}s", flush=True)
    R._build_fixed_pipeline = orig


if __name__ == "__main__":
    main()
