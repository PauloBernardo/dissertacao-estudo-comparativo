#!/usr/bin/env python3
"""FT-CUR: quanto o F1 depende de m/N? (o Nyström teve m na grade; o FT-CUR só 10–20%.)

Hiperparâmetros FIXOS = seleção per_batch do Tier 3 (tier3_selection_cur_gpu*_per.json);
varia só m_ratio. N = 20 mil sorteados do treino do Tier 3, teste = os mesmos 50 mil.
Lote 512 → ~39 passos por época (Tier 3: 200 mil / 4096 ≈ 49). per_batch: cada lote usa
max(2, round(m_ratio·512)) landmarks; a predição (streaming) usa m = m_ratio·N.

    .venv/bin/python scripts/study_ftcur_m_sweep.py
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import run_tier2_gridcv as g  # noqa: E402
import run_tier3 as t3  # noqa: E402

V = "FTTransformerCURRandom"
PARAMS = {  # seleção per_batch do Tier 3 (Kaggle, 2026-10-18)
    "HIGGS500K": {"n_layers": 1, "n_heads": 2, "lr": 0.0014691510255403506},
    "SUSY": {"n_layers": 1, "n_heads": 2, "lr": 0.00033873945602584123},
}
RATIOS = [0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(PARAMS))
    p.add_argument("--ratios", nargs="+", type=float, default=RATIOS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--n-train", type=int, default=20000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--output", default=str(ROOT / "results/study_ftcur_m_sweep.json"))
    a = p.parse_args()
    out = Path(a.output)
    recs = json.loads(out.read_text()) if out.exists() else []
    done = {(r["dataset"], r["m_ratio"], r["seed"]) for r in recs}
    fmt = g._label_format(V)
    for ds in a.datasets:
        for seed in a.seeds:
            X_tr, X_te, y_tr, y_te, _ = t3.load_split(ds, seed, fmt)
            X, _, y, _ = train_test_split(X_tr, y_tr, train_size=a.n_train,
                                          stratify=y_tr, random_state=seed)
            for r in a.ratios:
                if (ds, r, seed) in done:
                    continue
                g.set_global_seed(seed)
                pipe = t3.build(V, seed, {**PARAMS[ds], "m_ratio": r, "batch_size": a.batch,
                                          "predict_mode": "streaming"})
                t0 = time.perf_counter(); pipe.fit(X, y); fit_s = time.perf_counter() - t0
                f1 = t3._f1(y_te, pipe.predict(X_te), fmt)
                clf = pipe.named_steps["clf"]
                rec = {"dataset": ds, "m_ratio": r, "seed": seed, "n_train": len(X),
                       "m": int(round(r * len(X))), "m_per_batch": max(2, round(r * a.batch)),
                       "batch_size": a.batch, "test_f1_macro": f1, "fit_time_s": fit_s,
                       "best_epoch": getattr(clf, "best_epoch_", None), **PARAMS[ds]}
                recs.append(rec)
                out.write_text(json.dumps(recs, indent=1))
                print(f"{ds:10s} s{seed} m/N={r:<6} m={rec['m']:5d}  F1 {f1:.4f}  {fit_s:6.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
