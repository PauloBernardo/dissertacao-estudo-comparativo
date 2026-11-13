#!/usr/bin/env python3
"""Arrhythmia (UCI id 5; 452 × 279): ablação da atenção entre amostras + painel de referência.

É o regime (poucas amostras, muitos atributos) em que o artigo do SAINT (Somepalli et al., 2021,
"When to use intersample attention?") relata ganho da atenção entre amostras. Mesmo split e
pré-processamento de study_intersample_arrhythmia.py (70/30 estratificado; mediana do treino nos
faltantes; colunas constantes no treino removidas; padronização no treino).

Modelos (todos 200 épocas, sem parada antecipada, restaurando o melhor checkpoint de validação):
  SAINT (com MISA) / SAINT (sem MISA) — configuração do artigo para o Arrhythmia (Apêndice C):
      d = 8, h = 4; lr = 1e-4 (a do artigo; com 5e-4 a MISA colapsa nesta base) e lote 32
      (o artigo usa lotes menores nas bases com muitas colunas). Só a MISA muda entre os dois.
  FT-CUR (CUR, m = 0,2N) / FT-CUR (sem inter) / FT-CUR (inter densa) — inter_mode do wrapper.
  FT-Softmax / FT-Sparsemax — FT inter-atributos; mesmos hiperparâmetros, só a função de atenção.
  ADMM-Nyström (ℓ1) — GridSearchCV 3-fold; σ na escala de √p.
  XGBoost — referência.
--label odds usa a binarização do artigo do SAINT (versão ODDS: classes raras 3,4,5,7,8,9,14,15
contra o resto; 66 positivos) em vez de normal × qualquer arritmia.

    python scripts/study_arrhythmia_kaggle.py --seeds 0 1 2 3 4 --output results/arr_g0.json
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402
from src.data.preprocessing import make_splits  # noqa: E402
from src.models.ft_transformer_saint_wrapper import SAINTColnorm  # noqa: E402
from src.models.ft_transformer_cur_wrapper import FTTransformerCURColnorm  # noqa: E402
from src.models.transformers.ft_transformer import FTTransformer  # noqa: E402
from src.models.lssvm.primal.admm_nystrom import ADMMNystromLSSVM  # noqa: E402

DATA = ROOT / "data/raw/arrhythmia/arrhythmia.data"
EP = dict(epochs=200, patience=200)


ODDS_RARE = [3, 4, 5, 7, 8, 9, 14, 15]       # classes raras da versão ODDS (66 positivos, 14,6%)


def load_arrhythmia(label: str = "any"):
    raw = np.genfromtxt(DATA, delimiter=",", missing_values="?", filling_values=np.nan)
    X, cls = raw[:, :-1], raw[:, -1].astype(int)
    if label == "odds":                       # binarização do SAINT (ODDS/Stony Brook): raras × resto
        return X, np.isin(cls, ODDS_RARE).astype(int)
    return X, (cls != 1).astype(int)          # 1 = arritmia (qualquer classe ≠ normal)


def builders(seed: int, p: int, ep: int = 200) -> dict:
    s0 = np.sqrt(p)
    EP = dict(epochs=ep, patience=ep)
    ft = dict(random_state=seed, batch_size=256, max_epochs=ep, patience=ep)
    saint = dict(d_model=8, n_heads=4, n_layers=2, lr=1e-4, batch_size=32, random_state=seed, **EP)
    cur = dict(random_state=seed, selection_method="random", m_ratio=0.2, **EP)
    return {
        "SAINT (com MISA)":     lambda: SAINTColnorm(use_misa=True, **saint),
        "SAINT (sem MISA)":     lambda: SAINTColnorm(use_misa=False, **saint),
        "FT-CUR (CUR)":         lambda: FTTransformerCURColnorm(inter_mode="cur", **cur),
        "FT-CUR (sem inter)":   lambda: FTTransformerCURColnorm(inter_mode="none", **cur),
        "FT-CUR (inter densa)": lambda: FTTransformerCURColnorm(inter_mode="dense", **cur),
        "FT-Softmax":           lambda: FTTransformer(attention_type="softmax", **ft),
        "FT-Sparsemax":         lambda: FTTransformer(attention_type="sparsemax", **ft),
        "ADMM-Nyström (ℓ1)":    lambda: GridSearchCV(
            ADMMNystromLSSVM(m_ratio=0.30, rho="adaptive", max_iter=500, kernel_reg=True,
                             landmark_method="random", random_state=seed),
            {"sigma": [0.5 * s0, s0, 2 * s0], "tau": [1.0, 0.1, 0.02],
             "lambda_rel": [0.2, 0.05, 0.01, 0.001]},
            scoring="f1_macro", cv=StratifiedKFold(3, shuffle=True, random_state=seed), n_jobs=2),
        "XGBoost":              lambda: XGBClassifier(n_estimators=200, max_depth=3,
                                                      random_state=seed, verbosity=0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--models", nargs="*", default=None, help="subconjunto (nomes exatos)")
    ap.add_argument("--label", choices=["any", "odds"], default="any",
                    help="any = normal × qualquer arritmia; odds = classes raras × resto (artigo do SAINT)")
    ap.add_argument("--smoke", action="store_true", help="2 épocas, só para testar o pipeline")
    ap.add_argument("--output", default=str(ROOT / "results/study_arrhythmia_kaggle.json"))
    a = ap.parse_args()
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    recs = json.loads(out.read_text()) if out.exists() else []
    done = {(r["seed"], r["model"]) for r in recs if r.get("status") == "ok"}
    X, y = load_arrhythmia(a.label)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Arrhythmia {X.shape} | rótulo {a.label} ({int(y.sum())} positivos) | device {dev}", flush=True)
    for seed in a.seeds:
        Xtr, Xte, ytr, yte = make_splits(X, y, test_size=0.30, seed=seed)
        imp = SimpleImputer(strategy="median").fit(Xtr)
        Xtr, Xte = imp.transform(Xtr), imp.transform(Xte)
        keep = Xtr.std(0) > 0
        Xtr, Xte = Xtr[:, keep], Xte[:, keep]
        sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        for name, build in builders(seed, Xtr.shape[1], 2 if a.smoke else 200).items():
            if (a.models and name not in a.models) or (seed, name) in done:
                continue
            torch.manual_seed(seed); np.random.seed(seed)
            if dev == "cuda":
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            rec = {"seed": seed, "model": name, "n_train": len(ytr), "p": int(Xtr.shape[1])}
            t0 = time.perf_counter()
            try:
                clf = build().fit(Xtr, ytr)
                rec["fit_s"] = time.perf_counter() - t0
                pred = np.asarray(clf.predict(Xte))
                s = clf.predict_proba(Xte)[:, 1]
                rec.update(status="ok", test_f1_macro=float(f1_score(yte, pred, average="macro")),
                           test_auc=float(roc_auc_score(yte, s)))
                if name.startswith("ADMM"):
                    b = clf.best_estimator_
                    rec.update(best_params={k: float(v) for k, v in clf.best_params_.items()},
                               sparsity=float(b.sparsity_ratio_), n_active=int(b.n_support_))
                if hasattr(clf, "best_epoch_"):
                    rec["best_epoch"] = int(clf.best_epoch_)
            except Exception as e:                       # registra e segue (ex.: OOM)
                rec.update(status="error", error=f"{type(e).__name__}: {e}"[:300],
                           fit_s=time.perf_counter() - t0)
            if dev == "cuda":
                rec["peak_gpu_mib"] = torch.cuda.max_memory_allocated() / 2**20
            recs.append(rec)
            tmp = out.with_suffix(".tmp"); tmp.write_text(json.dumps(recs, indent=1)); tmp.replace(out)
            f1 = rec.get("test_f1_macro", float("nan"))
            print(f"  seed {seed} {name:22s} {rec['status']:5s} F1 {f1:.3f} "
                  f"AUC {rec.get('test_auc', float('nan')):.3f} {rec['fit_s']:6.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
