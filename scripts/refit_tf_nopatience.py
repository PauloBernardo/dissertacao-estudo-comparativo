#!/usr/bin/env python3
"""Refit dos Transformers com o vencedor publicado, 200 épocas SEM paciência (2026-10-14).

Protocolo da versão 1 (decisão de 2026-10-14): a SELEÇÃO continua a publicada (GridSearchCV com
200/16), e só o AJUSTE FINAL é refeito com orçamento completo -- 200 épocas, paciência desligada,
restaurando o checkpoint de melhor validação interna. Motivo: com paciência 16, o ajuste final parava
no platô inicial em boa parte das sementes (FT-CUR na Ablação A: 0,5-0,65 -> 0,93-0,99 sem
paciência; amostra do Tier 1 em results/study_tf_patience_sample.json: +0,25 no TWS e +0,42 no TWC).
Refazer a grade sem paciência custaria ~4x o Tier 1 inteiro; selecionar com orçamento reduzido e
treinar o vencedor com o completo é a lógica do successive halving / Hyperband.

Cada registro de --source é reproduzido no MESMO split (hold-out por semente; subamostragem do Tier 2
quando --n-train é dado) com os best_params (ou fixed_params, na Ablação D) dele. O registro de saída
é uma cópia do original com as métricas de teste, esparsidade e orçamento trocadas; guarda o F1
anterior em ``test_f1_macro_pat16`` e marca ``protocol = "refit_200_nopatience"``.

Uso:
  Tier 1 / Ablações A, B, C:  --source results/tier1_gridcv.json --output results/refit_tier1.json
  Tier 2 (N=2000):            --source results/tier2_transformers.json --n-train 2000 --output ...
  Ablação D (N=5000):         --source results/tier2_fixedparams_n5000_transformers.json --n-train 5000 ...
  [--datasets ...] [--seeds ...] — retomável (pula o que já está em --output).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_tier1_gridcv as R1  # noqa: E402
import run_tier2_gridcv as R2  # noqa: E402
from src.experiments.reproducibility import set_global_seed  # noqa: E402

TF = ["FTTransformer_softmax", "FTTransformer_topk", "FTTransformer_entmax", "FTTransformer_sparsemax",
      "SAINTColnorm", "FTTransformerCURRandom"]
EPOCHS = 200
for _R in (R1, R2):                       # paciência = teto: nunca dispara
    _R.BUDGET_OVERRIDE.update(epochs=EPOCHS, patience=EPOCHS)


def refit(src: dict, n_train: int | None) -> dict:
    variant, dataset, seed = src["variant"], src["dataset"], src["seed"]
    R = R1 if n_train is None else R2
    rec = {k: v for k, v in src.items() if k != "traceback"}
    rec.update(protocol="refit_200_nopatience", test_f1_macro_pat16=src.get("test_f1_macro"),
               fit_time_s_pat16=src.get("fit_time_s"))
    set_global_seed(seed)
    try:
        X, y, _ = R1.DatasetLoader.load(dataset)
        if n_train is not None:
            X, y = R2._subsample(X, y, round(n_train / 0.70), seed)
        Xtr, Xte, ytr, yte = R1.make_splits(X, y, test_size=0.30, seed=seed)
        fmt = R._label_format(variant)
        ytr, yte = R1._convert_labels(ytr, fmt), R1._convert_labels(yte, fmt)
        params = src.get("best_params") or src.get("fixed_params")
        pipe, _ = R._build_pipeline(variant, seed)
        pipe.set_params(**{f"clf__{k}": v for k, v in params.items()})
        t0 = time.perf_counter()
        pipe.fit(Xtr, ytr)
        fit_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        test = R1._compute_test_metrics(pipe, Xte, yte, fmt)
        pred_t = time.perf_counter() - t0
        clf = pipe.named_steps["clf"]
        rec.update(status="ok", fit_time_s=fit_t, predict_time_s=pred_t, **test,
                   **R1._collect_sparsity_metrics(clf, Xte), **R1._budget_attrs(clf))
    except Exception as exc:  # noqa: BLE001
        rec.update(status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--n-train", type=int, default=None, help="subamostragem do Tier 2 (2000) / Ablação D (5000)")
    ap.add_argument("--models", nargs="+", default=TF)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--log-level", default="WARNING")
    a = ap.parse_args()
    logging.basicConfig(level=a.log_level)

    todo = [r for r in json.loads(a.source.read_text())
            if r.get("status") == "ok" and r["variant"] in a.models
            and (a.datasets is None or r["dataset"] in a.datasets)
            and (a.seeds is None or r["seed"] in a.seeds)]
    out = json.loads(a.output.read_text()) if a.output.exists() else []
    done = {(r["variant"], r["dataset"], r["seed"]) for r in out if r.get("status") == "ok"}
    out = [r for r in out if r.get("status") == "ok"]
    todo = [r for r in todo if (r["variant"], r["dataset"], r["seed"]) not in done]
    print(f"{len(todo)} a refazer ({len(done)} já feitos) -> {a.output}", flush=True)
    for i, src in enumerate(todo, 1):
        rec = refit(src, a.n_train)
        out.append(rec)
        tmp = a.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, indent=1)); os.replace(tmp, a.output)
        if rec["status"] == "ok":
            print(f"[{i}/{len(todo)}] {rec['variant']:24s} {rec['dataset']:8s} s{rec['seed']:<2d} "
                  f"F1 {rec['test_f1_macro_pat16']:.3f} -> {rec['test_f1_macro']:.3f}  "
                  f"best_ep {rec.get('best_epoch')}  {rec['fit_time_s']:.0f}s", flush=True)
        else:
            print(f"[{i}/{len(todo)}] ERRO {rec['variant']} {rec['dataset']} s{rec['seed']}: {rec['error']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
