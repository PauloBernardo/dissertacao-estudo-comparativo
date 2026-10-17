"""Tier 3: curva F1 × esparsidade em λ_rel para ADMM-/FISTA-Nyström.

Com σ e τ da seleção (limites largos), varre λ_rel no mesmo hold-out da seleção
(semente 0, 160 mil / 40 mil). Regra declarada: o maior λ_rel cujo F1 fique a até
TOL do melhor da curva (lógica da regra "1-SE" do Lasso).

    .venv/bin/python scripts/tier3_l1_plateau.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_tier2_gridcv as g  # noqa: E402
import run_tier3 as t3  # noqa: E402

LAMBDAS = np.logspace(-5, np.log10(0.4), 12).tolist()
TOL = 0.005
SEL = ROOT / "results" / "tier3_selection_limites_largos.json"
OUT = ROOT / "results" / "tier3_l1_plateau.json"


def main() -> int:
    sel = json.loads(SEL.read_text())
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for v in ("ADMMNystromRandom", "FISTANystromRandom"):
        for d in t3.DATASETS:
            if out.get(v, {}).get(d, {}).get("chosen"):
                continue
            fmt = g._label_format(v)
            X_tr, _, y_tr, _, _ = t3.load_split(d, t3.SELECT_SEED, fmt)
            X_fit, X_val, y_fit, y_val = train_test_split(
                X_tr, y_tr, test_size=t3.VAL_SIZE, stratify=y_tr, random_state=t3.SELECT_SEED)
            base = {k: p for k, p in sel[v][d]["params"].items() if k != "lambda_rel"}
            curve = []
            for lam in LAMBDAS:
                g.set_global_seed(t3.SELECT_SEED)
                params = {**base, "lambda_rel": lam, **t3.fixed_budget(v, len(X_fit))}
                pipe = t3.build(v, t3.SELECT_SEED, params).fit(X_fit, y_fit)
                f1 = t3._f1(y_val, pipe.predict(X_val), fmt)
                sp = g._collect_sparsity(pipe.named_steps["clf"], X_val)
                curve.append({"lambda_rel": lam, "val_f1": f1, **sp})
                print(v, d, f"λ={lam:.2e} F1={f1:.4f}", {k: sp[k] for k in list(sp)[:3]}, flush=True)
            best = max(c["val_f1"] for c in curve)
            chosen = max((c for c in curve if c["val_f1"] >= best - TOL), key=lambda c: c["lambda_rel"])
            out.setdefault(v, {})[d] = {"base_params": base, "tol": TOL, "curve": curve,
                                        "chosen": {**base, "lambda_rel": chosen["lambda_rel"]},
                                        "chosen_val_f1": chosen["val_f1"], "best_val_f1": best}
            OUT.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
