"""Tier 3 (v2): N = 200 mil, seleção única por base + ajuste final por semente.

Protocolo (PLANO_FECHAMENTO, §"VERSÃO 2 — Tier 3"; decisões de 2026-10-16):
  * m = r = 2000 fixo (≈1% de N) nos Nyström e no PCP.
  * Fase ``select``: uma busca Optuna (TPE, semente fixa) por (modelo, base), num
    hold-out 80/20 do treino da semente 0. Mesmo número de trials para todos.
    LSSVM e XGBoost treinam num passo só, então o pruner Hyperband não se aplica
    aqui (só aos Transformers, no Kaggle).
  * σ é categórico (valores da grade da v1) para que o cache da fatoração do PCP
    e a matriz C dos Nyström possam ser reaproveitados entre trials; τ, γ, λ e os
    hiperparâmetros do XGBoost são contínuos (log).
  * Fase ``final``: para cada semente, subamostra 200 mil, split 80/20 e treina
    com os hiperparâmetros congelados (esquema da Ablação D).

Estudos Optuna persistidos em SQLite (retomáveis); resultados finais retomáveis
por (modelo, base, semente). Uso:

    .venv/bin/python scripts/run_tier3.py select --variants PCPLSSVm --datasets HIGGS500K
    .venv/bin/python scripts/run_tier3.py final  --variants PCPLSSVm --datasets HIGGS500K --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import resource
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import optuna  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

import run_tier2_gridcv as g  # noqa: E402
from src.models.lssvm.primal.pcp_lssvm import PCPLSSVm  # noqa: E402

N_TOTAL   = int(os.environ.get("TIER3_N_TOTAL", 200_000))  # menor só p/ teste de fumaça
TEST_SIZE = 0.20           # 160 mil de treino
VAL_SIZE  = 0.20           # hold-out de seleção dentro do treino da semente 0
M_FIXED   = 2000
N_TRIALS  = 20
SELECT_SEED = 0
DATASETS  = ["HIGGS500K", "COVER", "KDD99"]

# Prazo absoluto (unix ts) para o "Save & Run All" do Kaggle: depois dele não se
# inicia trial nem semente nova; o que já terminou fica salvo e a próxima sessão retoma.
DEADLINE = float(os.environ.get("TIER3_DEADLINE", "inf"))


def _past_deadline() -> bool:
    return time.time() >= DEADLINE


SELECTION_FILE = ROOT / "results" / "tier3_selection.json"
FINAL_FILE     = ROOT / "results" / "tier3_final.json"
STORAGE        = f"sqlite:///{ROOT / 'results' / 'tier3_optuna.db'}"

# Com r = 2000, cada fatoração do PCP ocupa N·r·8 bytes (≈2 GB em 128 mil
# amostras); a v1 guardava até 80. Três entradas cabem nos 19 GB da máquina.
PCPLSSVm._CHOL_CACHE_MAX = 3

SIGMA_NYS = [0.5, 1.5, 5.0]
SIGMA_PCP = [float((2.0 * 2.0 ** k) ** -0.5) for k in range(-9, 4)]  # Zhou (2016)


def suggest(variant: str, trial: optuna.Trial) -> dict:
    """Espaço de busca: mesmos limites da grade da v1, agora contínuos."""
    if variant == "NystromLSSVMRandom":
        return {"sigma": trial.suggest_categorical("sigma", SIGMA_NYS),
                "gamma": trial.suggest_float("gamma", 0.1, 100.0, log=True)}
    if variant in ("ADMMNystromRandom", "FISTANystromRandom"):
        return {"sigma": trial.suggest_categorical("sigma", SIGMA_NYS),
                "tau": trial.suggest_float("tau", 0.01, 10.0, log=True),
                "lambda_rel": trial.suggest_float("lambda_rel", 1e-3, 0.4, log=True)}
    if variant == "PCPLSSVm":
        return {"sigma": trial.suggest_categorical("sigma", SIGMA_PCP),
                "tau": trial.suggest_float("tau", 1e-6, 1.0, log=True)}
    if variant.startswith("FTTransformer_"):
        return {"num_blocks": trial.suggest_int("num_blocks", 2, 4),
                "num_heads": trial.suggest_categorical("num_heads", [2, 4]),
                "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True)}
    if variant in ("SAINTColnorm", "FTTransformerCURRandom"):
        return {"n_layers": trial.suggest_int("n_layers", 1, 3),
                "n_heads": trial.suggest_categorical("n_heads", [2, 4]),
                "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True)}
    if variant == "XGBoost":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 1000, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True)}
    raise KeyError(variant)


def fixed_budget(variant: str, n_train: int) -> dict:
    """m = r = M_FIXED; os Nyström recebem m como fração do treino."""
    if variant == "PCPLSSVm":
        return {"rank": M_FIXED}
    if "Nystrom" in variant or variant == "FTTransformerCURRandom":
        return {"m_ratio": M_FIXED / n_train}
    return {}


# Transformers: protocolo 200/16 da v1 com paciência também no ajuste final
# (restaura o melhor checkpoint). Sem pruner: a paciência já interrompe cada trial,
# e os laços de treino não expõem callback por época.
TF_BUDGET = {"epochs": 200, "patience": 16}


def load_split(dataset: str, seed: int, fmt: str):
    X, y, meta = g.DatasetLoader.load(dataset)
    X, y = g._subsample(X, y, N_TOTAL, seed)
    X_tr, X_te, y_tr, y_te = g.make_splits(X, y, test_size=TEST_SIZE, seed=seed)
    return X_tr, X_te, g._convert_labels(y_tr, fmt), g._convert_labels(y_te, fmt), meta


def build(variant: str, seed: int, params: dict):
    if g.GRIDS[variant].get("needs_gpu"):
        g.BUDGET_OVERRIDE.update(TF_BUDGET)
    pipe, _ = g._build_pipeline(variant, seed)
    pipe.set_params(**{f"clf__{k}": v for k, v in params.items()})
    return pipe


def _f1(y_true, y_pred, fmt: str) -> float:
    if fmt == "signed":
        y_true, y_pred = (y_true + 1) // 2, (y_pred + 1) // 2
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _load(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


# ── Fase 1: seleção ──────────────────────────────────────────────────────────

def select(variant: str, dataset: str, n_trials: int) -> dict:
    fmt = g._label_format(variant)
    X_tr, _, y_tr, _, _ = load_split(dataset, SELECT_SEED, fmt)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=VAL_SIZE, stratify=y_tr, random_state=SELECT_SEED)
    budget = fixed_budget(variant, len(X_fit))

    def objective(trial: optuna.Trial) -> float:
        params = {**suggest(variant, trial), **budget}
        g.set_global_seed(SELECT_SEED)
        t0 = time.perf_counter()
        pipe = build(variant, SELECT_SEED, params).fit(X_fit, y_fit)
        trial.set_user_attr("fit_s", round(time.perf_counter() - t0, 2))
        return _f1(y_val, pipe.predict(X_val), fmt)

    study = optuna.create_study(
        study_name=f"{variant}__{dataset}", storage=STORAGE, load_if_exists=True,
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SELECT_SEED))
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    def stop_at_deadline(study, _trial):
        if _past_deadline():
            study.stop()

    if done < n_trials and not _past_deadline():
        study.optimize(objective, n_trials=n_trials - done, callbacks=[stop_at_deadline])

    fits = [t.user_attrs.get("fit_s", 0.0) for t in study.trials]
    return {"params": study.best_params, "best_val_f1": study.best_value,
            "n_trials": len(study.trials), "complete": len(study.trials) >= n_trials, "total_fit_s": round(sum(fits), 1),
            "n_fit": len(X_fit), "n_val": len(X_val), "m_fixed": M_FIXED,
            "tf_budget": TF_BUDGET if g.GRIDS[variant].get("needs_gpu") else None}


# ── Fase 2: ajuste final por semente ─────────────────────────────────────────

def final_one(variant: str, dataset: str, seed: int, selection: dict) -> dict:
    fmt = g._label_format(variant)
    rec = {"variant": variant, "dataset": dataset, "seed": seed,
           "tuning": "optuna_single_selection", "m_fixed": M_FIXED}
    try:
        g.set_global_seed(seed)
        X_tr, X_te, y_tr, y_te, meta = load_split(dataset, seed, fmt)
        params = {**selection["params"], **fixed_budget(variant, len(X_tr))}
        pipe = build(variant, seed, params)
        t0 = time.perf_counter()
        pipe.fit(X_tr, y_tr)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        metrics = g._compute_test_metrics(pipe, X_te, y_te, fmt)
        rec.update({
            "status": "ok", "params": params, "n_train": len(X_tr), "n_test": len(X_te),
            "dataset_tier": meta.get("tier"), "fit_time_s": fit_s,
            "predict_time_s": time.perf_counter() - t0,
            "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
            **metrics, **g._collect_sparsity(pipe.named_steps["clf"], X_te),
            **g._budget_attrs(pipe.named_steps["clf"])})
    except Exception as exc:
        rec.update({"status": "error", "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()})
    return rec


def main() -> int:
    global SELECTION_FILE, FINAL_FILE, STORAGE
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["select", "final"])
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--n-trials", type=int, default=N_TRIALS)
    p.add_argument("--selection-file", type=Path, default=SELECTION_FILE)
    p.add_argument("--final-file", type=Path, default=FINAL_FILE)
    p.add_argument("--storage", default=STORAGE, help="Optuna; um arquivo por processo em paralelo")
    a = p.parse_args()
    SELECTION_FILE, FINAL_FILE, STORAGE = a.selection_file, a.final_file, a.storage
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    optuna.logging.set_verbosity(optuna.logging.INFO)

    if a.phase == "select":
        for v in a.variants:
            for d in a.datasets:
                if _past_deadline():
                    logging.warning("prazo atingido — parando; a próxima sessão retoma")
                    return 0
                res = select(v, d, a.n_trials)
                sel = _load(SELECTION_FILE, {})
                sel.setdefault(v, {})[d] = res
                SELECTION_FILE.write_text(json.dumps(sel, indent=1))
                logging.info("SELEÇÃO %s %s → %s", v, d, res)
        return 0

    if a.seeds is None:
        p.error("--seeds é obrigatório na fase final")
    sel = _load(SELECTION_FILE, {})
    for v in a.variants:
        for d in a.datasets:
            if d not in sel.get(v, {}) or not sel[v][d].get("complete", True):
                logging.error("seleção ausente ou incompleta para %s %s — rode a fase select", v, d)
                continue
            for s in a.seeds:
                if _past_deadline():
                    logging.warning("prazo atingido — parando; a próxima sessão retoma")
                    return 0
                recs = _load(FINAL_FILE, [])
                if any(r["variant"] == v and r["dataset"] == d and r["seed"] == s
                       and r.get("status") == "ok" for r in recs):
                    continue
                rec = final_one(v, d, s, sel[v][d])
                recs = [r for r in _load(FINAL_FILE, [])
                        if not (r["variant"] == v and r["dataset"] == d and r["seed"] == s)]
                recs.append(rec)
                FINAL_FILE.write_text(json.dumps(recs, indent=1))
                logging.info("FINAL %s %s s%d → %s F1=%s", v, d, s, rec["status"],
                             rec.get("test_f1_macro"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
