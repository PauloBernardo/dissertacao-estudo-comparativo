#!/usr/bin/env python3
"""Benchmark completo para atualizar a Tabela 18 (família LSSVM) da dissertação.

Duas extensões sobre scripts/run_scaling_benchmark.py (o gerador atual):

1. FIT e PREDIÇÃO medidos SEPARADAMENTE (RAM de pico por fase). A Tabela 18
   publicada só tem uma coluna "Mem." — sem essa, é impossível ver se a
   esparsidade de fato reduz o custo da predição (que é onde ela deveria
   aparecer: menos vetores de suporte => kernel(teste, suporte) menor).
   Verificado: `_kernel_predict` (src/models/lssvm/base.py) FAZ subsetting
   real (`self.X_train_[nz]`, só os vetores de suporte não-nulos) -- ao
   contrário da esparsidade de atenção dos Transformers (numérica, não
   algorítmica), aqui a hipótese é genuinamente testável.

2. Adiciona os 4 modelos de maior esparsidade da Tabela 17 (PCP, IP,
   Pruning, OppMaps), ausentes da Tabela 18 -- o script anterior só media
   as 5 variantes ADMM/Nyström/denso. Hiperparâmetros VALIDADOS contra a
   Tabela 17 (moda das configs vencedoras no Tier 2 real,
   results/tier2_gridcv.json) -- não usa nada de OLD/scripts/ (benchmark
   anterior tinha PCP com sparsity_ratio=0.000, config errada/obsoleta):
       PCP:     sigma=5.0, tau=0.1,  rank=200            -> sparsity ~0.907
       IP:      sigma=5.0, tau=12.5, selection_ratio=0.7 -> sparsity ~0.518
       Pruning: sigma=5.0, tau=12.5                      -> sparsity ~0.542
       OppMaps: sigma=5.0, tau=2.5,  proto_ratio=0.7     -> sparsity ~0.839

Modelos CPU (needs_gpu=False no grids.py) -- não há OOM de GPU para
capturar erro automaticamente; os N_MAX por modelo abaixo são cautelosos,
calibrados por tempo observado, não por captura de exceção.

Uso (rodar de dentro do repo clonado, ou apontando --repo-root):
    python run_table18_benchmark.py --output table18_results.json

Acha a raiz do repo sozinho (procura src/models/lssvm/base.py subindo a
partir de onde está); se não achar, passe --repo-root.
"""
from __future__ import annotations

import argparse
from sklearn.metrics import f1_score
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("table18_bench")


def _find_repo_root(start: Path, marker: str = "src/models/lssvm/base.py") -> Path:
    cur = start.resolve()
    for _ in range(6):
        if (cur / marker).exists():
            return cur
        cur = cur.parent
    raise SystemExit(
        f"Não achei a raiz do repo (procurei por '{marker}' subindo a partir de {start}). "
        "Rode com --repo-root apontando para a pasta que contém src/."
    )


def _setup_paths(repo_root: str | None) -> None:
    root = Path(repo_root) if repo_root else _find_repo_root(Path(__file__).parent)
    sys.path.insert(0, str(root))
    log.info("Repo root: %s", root)


# ── Configuração ──────────────────────────────────────────────────────────────

N_VALUES = [500, 1000, 2000, 5000, 10000, 20000]
N_TEST = 500

# N máximo por modelo. StandardLSSVM/ADMMNesterov são O(N³) no fit (herdado
# do script original). Os 4 novos: caps conservadores até calibrar tempo
# real -- PCP usa Cholesky pivotada (deve escalar bem, O(N·rank²)); IP monta
# o "sistema cheio" antes de podar (risco O(N²)-O(N³)); Pruning treina CG
# denso + retreina várias vezes; OppMaps faz K-means no espaço de kernel
# (O(N²) por iteração). Calibrado empiricamente (não é chute):
#   PCP:      1,6s em N=5000  -> cresce bem, mantém até 20000
#   OppMaps:  9,6s em N=5000  -> ~O(N^2.1), ainda OK até 20000 (~9min/rep)
#   IP:       11,6s em N=2000 -> ~O(N^2.7); N=5000 já ~150s/rep, parar aqui
#   Pruning:  23,3s em N=2000 -> ~O(N^2.5); N=5000 já ~230s/rep, parar aqui
# IP e Pruning SEM checkpoint intra-N (só salva depois das `repeats`
# completarem) -- deixar N_MAX baixo evita ficar 1h+ sem salvar nada.
N_MAX = {
    "StandardLSSVM":       5000,
    "ADMMNesterovLSSVM":   5000,
    "NystromLSSVMColnorm": 20000,
    "ADMMNystromLSSVM":    20000,
    "FISTANystromLSSVM":   20000,
    "PCPLSSVm":            20000,
    "IPLSSVmOriginal":     5000,
    "PruningLSSVM":        5000,
    "OppositeMapsOriginalLSSVM": 20000,
    "DualFISTA":           5000,
}

_PROC = psutil.Process(os.getpid())


def _make_clf(variant: str, seed: int):
    if variant == "StandardLSSVM":
        from src.models.lssvm.standard import StandardLSSVM
        return StandardLSSVM(sigma=5.0, tau=2.5)  # σ=0,5 previa uma classe só
    elif variant == "ADMMNesterovLSSVM":
        from src.models.lssvm.primal.admm_nesterov import ADMMNesterovLSSVM
        return ADMMNesterovLSSVM(sigma=5.0, tau=0.5, lambda_=0.01, rho=None, max_iter=500)  # σ=0,5 previa uma classe só
    elif variant == "NystromLSSVMColnorm":
        # 2026-10-14: seletor random (decisão de 2026-09-19); a chave da variante fica a antiga.
        from src.models.nystrom_lssvm_wrapper import NystromLSSVMRandom
        return NystromLSSVMRandom(sigma=5.0, gamma=10.0, m_ratio=float(os.environ.get("M_RATIO", "0.30")), random_state=seed)  # σ=0,5 previa uma classe só
    elif variant == "ADMMNystromLSSVM":
        from src.models.lssvm.primal.admm_nystrom import ADMMNystromLSSVM
        # 2026-10-15: formulação com ½θᵀWθ; σ, τ e ℓ1 pela moda do Tier 2 (σ=0,5 previa uma classe só; esparsidade 0,89).
        return ADMMNystromLSSVM(sigma=5.0, tau=0.1, lambda_rel=0.001, kernel_reg=True,
                                m_ratio=float(os.environ.get("M_RATIO", "0.30")), landmark_method="random",
                                rho="adaptive", max_iter=500, random_state=seed)
    elif variant == "FISTANystromLSSVM":
        from src.models.lssvm.primal.fista_nystrom import FISTANystromLSSVM
        return FISTANystromLSSVM(sigma=5.0, tau=0.1, lambda_rel=0.001, kernel_reg=True,
                                 m_ratio=float(os.environ.get("M_RATIO", "0.30")), landmark_method="random",
                                 max_iter=5000, random_state=seed)
    # ── Modelos novos (ausentes da Tabela 18), config validada c/ Tabela 17 ──
    elif variant == "PCPLSSVm":
        from src.models.lssvm.primal.pcp_lssvm import PCPLSSVm
        return PCPLSSVm(sigma=5.0, tau=0.1, rank=200)
    elif variant == "IPLSSVmOriginal":
        from src.models.lssvm.dual.ip_lssvm_original import IPLSSVmOriginal
        return IPLSSVmOriginal(sigma=5.0, tau=12.5, selection_ratio=0.7)
    elif variant == "PruningLSSVM":
        from src.models.lssvm.dual.p_lssvm import PruningLSSVM
        return PruningLSSVM(sigma=5.0, tau=12.5, drop_tolerance=0.05, pruning_rate=0.05)
    elif variant == "OppositeMapsOriginalLSSVM":
        from src.models.lssvm.dual.opposite_maps_original import OppositeMapsOriginalLSSVM
        return OppositeMapsOriginalLSSVM(sigma=5.0, tau=2.5, proto_ratio=0.7, random_state=seed)
    elif variant == "DualFISTA":
        # config = moda das vencedoras no Tier 2 (97/180 regs, sigma=2.0,
        # tau=1.0, lambda=0.01). No dado sintético deste benchmark essa
        # combinação produz esparsidade baixa (~1-2%, não ~4,5% -- o 4,5%
        # do texto é a MÉDIA ponderada de todas as configs vencedoras do
        # Tier 2 em datasets reais, não desta config isolada em dado
        # sintético; mesma discrepância já observada para IPLSSVmOriginal).
        # decision_function usa _kernel_predict com subsetting -- sem bug,
        # só nunca esteve nesta tabela.
        from src.models.lssvm.dual.fista_dual_lssvm import DualFISTALSSVM
        return DualFISTALSSVM(sigma=2.0, tau=1.0, lambda_=0.01)
    raise ValueError(variant)


_REAL_CACHE: dict = {}
N_TEST_REAL = 2000


def make_data(n: int, seed: int) -> tuple:
    # 2026-10-15: DATASET=adult|credit|bank usa uma base real do Tier 2 (teste estratificado de 2000, treino de n).
    ds = os.environ.get("DATASET")
    if ds:
        from sklearn.model_selection import train_test_split
        if ds not in _REAL_CACHE:
            from src.data import loaders as L
            _REAL_CACHE[ds] = getattr(L, "_load_" + ds)()[:2]
        X, y = _REAL_CACHE[ds]
        X_rest, X_te, y_rest, y_te = train_test_split(X, y, test_size=N_TEST_REAL, stratify=y, random_state=seed)
        X_tr, _, y_tr, _ = train_test_split(X_rest, y_rest, train_size=n, stratify=y_rest, random_state=seed)
        return X_tr, X_te, 2 * y_tr - 1, 2 * y_te - 1
    X, y = make_classification(
        n_samples=n + N_TEST, n_features=20, n_informative=10, n_redundant=5,
        random_state=seed,
    )
    y = 2 * y - 1  # 2026-10-15: rótulos ±1, convenção dos modelos (com 0/1 o Nyström-SVM previa uma classe só)
    return X[:n], X[n:], y[:n], y[n:]


def _monitor_rss(stop_event: threading.Event, peak_mb: list) -> None:
    while not stop_event.is_set():
        try:
            rss = _PROC.memory_info().rss / 1024 / 1024
            if rss > peak_mb[0]:
                peak_mb[0] = rss
        except psutil.NoSuchProcess:
            break
        time.sleep(0.01)


def _run_phase(fn):
    """Executa fn() medindo tempo (s) e pico de RAM ADICIONAL (MB)."""
    baseline_mb = _PROC.memory_info().rss / 1024 / 1024
    peak_mb = [baseline_mb]
    stop = threading.Event()
    mon = threading.Thread(target=_monitor_rss, args=(stop, peak_mb), daemon=True)
    mon.start()
    t0 = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - t0
    stop.set()
    mon.join()
    return elapsed, max(peak_mb[0] - baseline_mb, 0.0)


def _measure(variant: str, X_tr, y_tr, X_te, seed: int, y_te=None) -> dict:
    clf = _make_clf(variant, seed)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    fit_s, fit_ram = _run_phase(lambda: pipe.fit(X_tr, y_tr))

    pred_holder = {}
    pred_s, pred_ram = _run_phase(lambda: pred_holder.__setitem__("y", pipe.predict(X_te)))

    inner = pipe.named_steps["clf"]
    sparsity = getattr(inner, "sparsity_ratio_", None)
    n_support = getattr(inner, "n_support_", None)

    return {
        "fit_s": round(fit_s, 3),
        "fit_ram_mb": round(fit_ram, 1),
        "pred_s": round(pred_s, 4),
        "pred_ram_mb": round(pred_ram, 1),
        "sparsity_ratio": float(sparsity) if sparsity is not None else None,
        "f1": float(f1_score(y_te, pred_holder["y"], average="macro")) if y_te is not None else None,
        "n_support": int(n_support) if n_support is not None else None,
        "n_iter": getattr(inner, "n_iter_", None),
        "converged": getattr(inner, "converged_", None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="table18_results.json")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Repetições por (modelo, N) para mediana estável")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--variants", default=None,
                        help="Lista separada por vírgula p/ rodar só um subconjunto. Default: todas.")
    args = parser.parse_args()

    _setup_paths(args.repo_root)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resumível.
    results: list[dict] = json.loads(out_path.read_text()) if out_path.exists() else []
    done_keys = {(r["variant"], r["n"]) for r in results}
    if results:
        log.info("Retomando de %s: %d combinações já registradas.", out_path, len(done_keys))

    variants = [
        "StandardLSSVM", "ADMMNesterovLSSVM",
        "NystromLSSVMColnorm", "ADMMNystromLSSVM", "FISTANystromLSSVM",
        "PCPLSSVm", "IPLSSVmOriginal", "PruningLSSVM", "OppositeMapsOriginalLSSVM",
        "DualFISTA",
    ]
    if args.variants:
        wanted = [v.strip() for v in args.variants.split(",") if v.strip()]
        unknown = [v for v in wanted if v not in variants]
        if unknown:
            raise SystemExit(f"Variantes desconhecidas: {unknown}\nDisponíveis: {variants}")
        variants = wanted
        log.info("Rodando apenas: %s", variants)

    for variant in variants:
        log.info("=== %s ===", variant)
        max_n = N_MAX[variant]

        for n in N_VALUES:
            if (variant, n) in done_keys:
                continue
            if n > max_n:
                results.append({"variant": variant, "n": n, "skipped": True, "reason": "acima do N_MAX cauteloso"})
                done_keys.add((variant, n))
                continue

            fits, fit_rams, preds, pred_rams, sps, nsups, f1s = [], [], [], [], [], [], []
            iters, convs = [], []
            last_error = None
            for rep in range(args.repeats):
                try:
                    X_tr, X_te, y_tr, y_te = make_data(n, seed=rep)
                    m = _measure(variant, X_tr, y_tr, X_te, seed=rep, y_te=y_te)
                    fits.append(m["fit_s"]); fit_rams.append(m["fit_ram_mb"])
                    preds.append(m["pred_s"]); pred_rams.append(m["pred_ram_mb"])
                    if m["sparsity_ratio"] is not None:
                        sps.append(m["sparsity_ratio"])
                    if m["f1"] is not None:
                        f1s.append(m["f1"])
                    if m["n_support"] is not None:
                        nsups.append(m["n_support"])
                    if m["n_iter"] is not None:
                        iters.append(int(m["n_iter"])); convs.append(bool(m["converged"]))
                    log.info("  N=%6d rep=%d  fit=%.2fs(%.0fMB)  pred=%.3fs(%.0fMB)  sparsity=%s",
                             n, rep, m["fit_s"], m["fit_ram_mb"], m["pred_s"], m["pred_ram_mb"],
                             f"{m['sparsity_ratio']:.3f}" if m["sparsity_ratio"] is not None else "-")
                except Exception as e:
                    last_error = str(e)
                    log.warning("  N=%6d rep=%d ERRO: %s", n, rep, e)
                    break

            if not fits:
                results.append({"variant": variant, "n": n, "skipped": True, "error": last_error})
                done_keys.add((variant, n))
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2)
                continue

            done_keys.add((variant, n))
            results.append({
                "variant": variant, "n": n, "skipped": False,
                "fit_s_median":     round(float(np.median(fits)), 3),
                "fit_ram_mb_median": round(float(np.median(fit_rams)), 1),
                "pred_s_median":    round(float(np.median(preds)), 4),
                "pred_ram_mb_median": round(float(np.median(pred_rams)), 1),
                "sparsity_ratio_median": round(float(np.median(sps)), 4) if sps else None,
                "n_support_median": round(float(np.median(nsups)), 1) if nsups else None,
                "f1_median": round(float(np.median(f1s)), 4) if f1s else None,
                "n_iter_median": float(np.median(iters)) if iters else None,
                "converged_frac": float(np.mean(convs)) if convs else None,
            })
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Salvo em %s", out_path)


if __name__ == "__main__":
    main()
