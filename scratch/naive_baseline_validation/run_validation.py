#!/usr/bin/env python3
"""Metodologia de validacao: para CADA modelo esparso do estudo, no ponto
de operacao que ele REALMENTE atingiu no GridSearchCV oficial (n_support
real, nao um k escolhido por nos), compara com um baseline "esparso
ingenuo" -- LSSVM denso treinado direto em k = n_support pontos
amostrados do MESMO pool de treino, na MESMA seed (mesmo split de teste).

Nao re-treina os outros modelos -- reaproveita test_f1_macro e
n_support_vectors ja gravados em results/tier1_gridcv.json e
results/tier2_gridcv.json. So o baseline ingenuo e novo.

Reconstroi o pool/split EXATAMENTE como o protocolo oficial (mesma
funcao, mesma seed -- garante o mesmo X_test bit-a-bit):
  - Tier1 (scripts/run_tier1_gridcv.py): sem subsample, make_splits(X, y,
    test_size=0.30, seed=seed).
  - Tier2 (scripts/run_tier2_gridcv.py): cap=round(2000/0.70)=2857,
    StratifiedShuffleSplit(train_size=cap, seed), depois make_splits(...,
    test_size=0.30, seed=seed).

Pula: StandardLSSVM (e o proprio denso), Transformers/XGBoost (esparsidade
de atencao ou nenhuma, nao amostral), e registros com n_support=0 ou
n_support=n_train (colapso total/nenhuma compressao -- nao ha k
informativo pra comparar).

Resumivel via JSON.

Uso:
    python run_validation.py [--tier tier1 tier2] [--output validation_results.json]
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import argparse


def _find_repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / "src" / "models" / "ft_transformer_model.py").exists():
            return parent
    raise RuntimeError("Repo root nao encontrado")


ROOT = _find_repo_root()
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.data.loaders import DatasetLoader
from src.data.preprocessing import _convert_labels, make_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("validation")

SKIP_VARIANTS = {"StandardLSSVM", "XGBoost", "FTTransformer_softmax", "FTTransformer_topk",
                  "FTTransformer_entmax", "FTTransformer_sparsemax", "FTTransformerCURColnorm",
                  "FTTransformerCURRandom",
                  "SAINTColnorm"}

# StandardLSSVM (sigma, tau) por base -- moda das vencedoras nos GridSearchCV
# oficiais (results/tier1_gridcv.json / tier2_gridcv.json). Mesmos valores
# usados em scratch/memory_matched_curve/run_memory_matched_curve.py.
STD_HYPERPARAMS = {
    "BCW": {"sigma": 5.0, "tau": 12.5}, "PID": {"sigma": 5.0, "tau": 2.5},
    "HAB": {"sigma": 1.5, "tau": 2.5}, "VCP": {"sigma": 5.0, "tau": 50.0},
    "GCR": {"sigma": 5.0, "tau": 2.5}, "AUS": {"sigma": 5.0, "tau": 0.5},
    "AI4I": {"sigma": 1.5, "tau": 12.5}, "TWS": {"sigma": 0.5, "tau": 12.5},
    "TWM": {"sigma": 0.15, "tau": 0.1}, "TWC": {"sigma": 0.5, "tau": 12.5},
    "ADULT": {"sigma": 5.0, "tau": 12.5}, "BANK": {"sigma": 5.0, "tau": 50.0},
    "CREDIT": {"sigma": 5.0, "tau": 12.5}, "HIGGS50K": {"sigma": 5.0, "tau": 0.5},
    "SHOPPERS": {"sigma": 5.0, "tau": 50.0}, "TELCO": {"sigma": 5.0, "tau": 0.5},
}

N_TRAIN_TIER2 = 2000
TEST_SIZE = 0.30


def _key(variant: str, dataset: str, seed: int) -> str:
    return f"{variant}__{dataset}__seed{seed}"


def _load_done(path: Path) -> tuple[list[dict], set[str]]:
    if not path.exists():
        return [], set()
    records = json.loads(path.read_text())
    done = {_key(r["variant"], r["dataset"], r["seed"]) for r in records if r.get("status") == "ok"}
    return records, done


def _save(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(records, indent=2, default=str))
    tmp.replace(path)


def _sample_k(X, y_raw, k: int, seed: int):
    """k pontos estratificados do pool; cai para amostragem simples (sem
    estratificar) quando k esta perto demais do tamanho do pool pro
    StratifiedShuffleSplit conseguir formar o complemento (min 1 por
    classe do lado de fora)."""
    n = len(X)
    n_classes = len(np.unique(y_raw))
    if n - k >= n_classes:
        sss_k = StratifiedShuffleSplit(n_splits=1, train_size=k, random_state=seed)
        idx, _ = next(sss_k.split(X, y_raw))
    else:
        idx = np.random.RandomState(seed).choice(n, size=k, replace=False)
    return idx


def _f1(y_true_signed, y_pred_signed) -> float:
    y_true_e = ((y_true_signed + 1) // 2).astype(int)
    y_pred_e = ((y_pred_signed + 1) // 2).astype(int)
    return float(f1_score(y_true_e, y_pred_e, average="macro", zero_division=0))


def _reconstruct_pool_test(dataset: str, tier: str, seed: int):
    X_full, y_full, _meta = DatasetLoader.load(dataset)
    if tier == "tier1":
        X_sub, y_sub = X_full, y_full
    else:
        n_total_cap = round(N_TRAIN_TIER2 / 0.70)
        if len(X_full) > n_total_cap:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=n_total_cap, random_state=seed)
            idx, _ = next(sss.split(X_full, y_full))
            X_sub, y_sub = X_full[idx], y_full[idx]
        else:
            X_sub, y_sub = X_full, y_full

    X_train, X_test, y_train_raw, y_test_raw = make_splits(X_sub, y_sub, test_size=TEST_SIZE, seed=seed)
    return X_train, X_test, y_train_raw, y_test_raw


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tiers", nargs="+", default=["tier1", "tier2"])
    p.add_argument("--output", type=Path, default=Path(__file__).parent / "validation_results.json")
    args = p.parse_args()

    from src.models.lssvm.standard import StandardLSSVM

    tier1_recs = json.loads((ROOT / "results" / "tier1_gridcv.json").read_text())
    tier2_recs = json.loads((ROOT / "results" / "tier2_gridcv.json").read_text())

    candidates: list[dict] = []
    if "tier1" in args.tiers:
        candidates += [{**r, "tier": "tier1"} for r in tier1_recs]
    if "tier2" in args.tiers:
        candidates += [{**r, "tier": "tier2"} for r in tier2_recs]

    candidates = [
        r for r in candidates
        if r.get("status") == "ok"
        and r["variant"] not in SKIP_VARIANTS
        and r.get("n_support_vectors") is not None
        and 0 < r["n_support_vectors"] < r["n_train"]
    ]
    logger.info("Candidatos (modelo, base, seed) com k informativo: %d", len(candidates))

    records, done = _load_done(args.output)
    logger.info("Retomando de %s: %d entradas ja registradas.", args.output, len(records))

    cache: dict[tuple, Any] = {}  # (dataset, tier, seed) -> (X_train_s, X_test_s, y_train, y_train_raw, y_test)

    for i, r in enumerate(candidates, start=1):
        variant, dataset, seed, tier = r["variant"], r["dataset"], r["seed"], r["tier"]
        k = int(r["n_support_vectors"])
        key = _key(variant, dataset, seed)
        if key in done:
            continue

        cache_key = (dataset, tier, seed)
        if cache_key not in cache:
            X_train, X_test, y_train_raw, y_test_raw = _reconstruct_pool_test(dataset, tier, seed)
            scaler = StandardScaler().fit(X_train)
            X_train_s = scaler.transform(X_train)
            X_test_s = scaler.transform(X_test)
            y_train = _convert_labels(y_train_raw, "signed")
            y_test = _convert_labels(y_test_raw, "signed")
            cache[cache_key] = (X_train_s, X_test_s, y_train_raw, y_train, y_test)
            if len(cache) > 50:
                cache.pop(next(iter(cache)))
        X_train_s, X_test_s, y_train_raw, y_train, y_test = cache[cache_key]

        record: dict[str, Any] = {
            "variant": variant, "dataset": dataset, "tier": tier, "seed": seed,
            "k": k, "n_train_pool": len(X_train_s),
            "f1_official": r["test_f1_macro"], "sparsity_ratio_official": r.get("sparsity_ratio"),
        }
        try:
            k_idx = _sample_k(X_train_s, y_train_raw, k, seed)
            X_k, y_k = X_train_s[k_idx], y_train[k_idx]

            t0 = time.perf_counter()
            model = StandardLSSVM(**STD_HYPERPARAMS[dataset])
            model.fit(X_k, y_k)
            fit_time = time.perf_counter() - t0
            f1_naive = _f1(y_test, model.predict(X_test_s))

            record.update({"status": "ok", "f1_naive": f1_naive, "fit_s_naive": fit_time})
        except Exception as exc:
            record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            logger.error("  %s/%s/seed%d/k%d FALHOU: %s", variant, dataset, seed, k, exc)

        records.append(record)
        done.add(key)
        if i % 25 == 0:
            _save(args.output, records)
            logger.info("[%d/%d] ... (%s/%s/seed%d, k=%d)", i, len(candidates), variant, dataset, seed, k)

    _save(args.output, records)
    logger.info("Concluido. Total: %d -> %s", len(records), args.output)


if __name__ == "__main__":
    main()
