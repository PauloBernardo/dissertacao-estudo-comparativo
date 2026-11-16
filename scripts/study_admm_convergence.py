#!/usr/bin/env python3
"""Convergência do ADMM-Nesterov nos modelos finais do Cenário 1: limite de 500 (publicado) contra 5000 iterações."""
import json, sys, logging, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
logging.disable(logging.WARNING)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.data.loaders import DatasetLoader
from src.data.preprocessing import _convert_labels, make_splits
from src.experiments.runner import _build_model
from src.tuning.grids import GRIDS
from run_tier1_gridcv import _compute_test_metrics
var = "ADMMNesterovLSSVM"
pub = {(r["dataset"], r["seed"]): r for r in json.loads((ROOT/"results/tier1_gridcv.json").read_text())
       if r["variant"] == var and r.get("status") == "ok"}
out = []
for ds in ["AI4I","AUS","BCW","GCR","HAB","PID","TWC","TWM","TWS","VCP"]:
    X, y, _ = DatasetLoader.load(ds)
    for seed in range(10):
        Xtr, Xte, a, b = make_splits(X, y, test_size=0.30, seed=seed)
        ytr, yte = _convert_labels(a, "signed"), _convert_labels(b, "signed")
        bp = pub[(ds, seed)]["best_params"]
        row = {"dataset": ds, "seed": seed, "pub_f1": pub[(ds, seed)]["test_f1_macro"]}
        for mi in [500, 5000]:
            params = {**GRIDS[var]["fixed"], **bp, "max_iter": mi}
            est, _ = _build_model(GRIDS[var]["model_name"], params, label_format="signed")
            pipe = Pipeline([("s", StandardScaler()), ("clf", est)]).fit(Xtr, ytr)
            c = pipe.named_steps["clf"]
            m = _compute_test_metrics(pipe, Xte, yte, "signed")
            row[mi] = {"conv": bool(c.converged_), "it": int(c.n_iter_), "f1": m["test_f1_macro"],
                       "sp": float(c.sparsity_ratio_)}
        out.append(row)
    print(ds, flush=True)
Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "results/study_admm_convergence.json").write_text(json.dumps(out, indent=1))
