#!/usr/bin/env python3
"""ν = 0 (sem Tikhonov) nos primais ℓ1 densos, hiperparâmetros publicados; registra falha do Cholesky de G e o reforço automático de 1e-3."""
import json, sys, logging, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
class H(logging.Handler):
    n = 0
    def emit(self, r):
        if "Cholesky failed" in r.getMessage(): H.n += 1
logging.getLogger().addHandler(H()); logging.getLogger().setLevel(logging.WARNING)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.data.loaders import DatasetLoader
from src.data.preprocessing import _convert_labels, make_splits
from src.experiments.runner import _build_model
from src.tuning.grids import GRIDS
from run_tier1_gridcv import _compute_test_metrics
F = {(r["variant"], r["dataset"], r["seed"]): r for r in json.loads((ROOT/"results/study_tikhonov_sensitivity.json").read_text()) if r["nu"] == 1e-2}
out = []
for ds in ["AI4I","AUS","BCW","GCR","HAB","PID","TWC","TWM","TWS","VCP"]:
    X, y, _ = DatasetLoader.load(ds)
    for seed in range(10):
        Xtr, Xte, a, b = make_splits(X, y, test_size=0.30, seed=seed)
        ytr, yte = _convert_labels(a, "signed"), _convert_labels(b, "signed")
        Xs = StandardScaler().fit(Xtr).transform(Xtr)
        for var in ["ADMMNesterovLSSVM", "FISTANesterov"]:
            ref = F[(var, ds, seed)]; bp = ref["best_params"]
            d2 = (Xs**2).sum(1)[:, None] + (Xs**2).sum(1)[None] - 2 * Xs @ Xs.T
            G = np.exp(-np.maximum(d2, 0) / (2 * bp["sigma"] ** 2))
            try: np.linalg.cholesky(G); chol = True
            except np.linalg.LinAlgError: chol = False
            H.n = 0
            params = {**GRIDS[var]["fixed"], **bp, "tikhonov_reg": 0.0}
            est, _ = _build_model(GRIDS[var]["model_name"], params, label_format="signed")
            pipe = Pipeline([("s", StandardScaler()), ("clf", est)])
            try:
                pipe.fit(Xtr, ytr); m = _compute_test_metrics(pipe, Xte, yte, "signed")
                f1, sp, err = m["test_f1_macro"], float(pipe.named_steps["clf"].sparsity_ratio_), None
            except Exception as e:
                f1, sp, err = None, None, repr(e)[:120]
            out.append({"variant": var, "dataset": ds, "seed": seed, "chol_G_ok": chol, "fallback": H.n > 0,
                        "f1": f1, "sp": sp, "err": err, "f1_nu001": ref["test_f1_macro"], "sp_nu001": ref["sparsity_ratio"],
                        "lambda_min": ref["lambda_min"]})
    print(ds, flush=True)
Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "results/study_tikhonov_nu0.json").write_text(json.dumps(out, indent=1))
