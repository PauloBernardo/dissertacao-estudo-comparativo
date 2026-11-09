#!/usr/bin/env python3
"""Ablação da atenção entre instâncias no Arrhythmia (UCI id 5): poucas amostras, muitos atributos.

É o regime em que o artigo do SAINT (Somepalli et al., 2021, Sec. "When to use intersample
attention?") relata ganho da atenção entre amostras (Arrhythmia, Arcene). Binarização usual:
classe 1 (ECG normal) contra as demais. Faltantes ('?') imputados pela mediana do treino; colunas
constantes no treino removidas; padronização ajustada no treino. Split estratificado 70/30.
Modelos: SAINT com/sem MISA, FT-CUR com/sem atenção inter-instâncias (CUR, m = N), FT-CUR com
atenção inter-instâncias densa, FT-Softmax e XGBoost. Hiperparâmetros padrão, 200 épocas sem
parada antecipada (estudo complementar; não reproduz o protocolo do Cenário 1).

    .venv/bin/python scripts/study_intersample_arrhythmia.py
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
from sklearn.preprocessing import StandardScaler  # noqa: E402
from src.data.preprocessing import make_splits  # noqa: E402
import study_intersample_synthetic as S  # noqa: E402
from study_ftcur_dense_inter import FTCURDense  # noqa: E402


def load_arrhythmia():
    raw = np.genfromtxt(ROOT / "data/raw/arrhythmia/arrhythmia.data", delimiter=",",
                        missing_values="?", filling_values=np.nan)
    X, cls = raw[:, :-1], raw[:, -1].astype(int)
    return X, (cls != 1).astype(int)          # 1 = arritmia (qualquer classe ≠ normal)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--output", default=str(ROOT / "results/study_intersample_arrhythmia.json"))
    p.add_argument("--skip", nargs="*", default=["SAINT"], help="prefixos de modelos a pular (SAINT não cabe em memória com p=279)")
    a = p.parse_args()
    out = Path(a.output)
    recs = json.loads(out.read_text()) if out.exists() else []
    done = {(r["seed"], r["model"]) for r in recs}
    X, y = load_arrhythmia()
    print(f"Arrhythmia: {X.shape}, positivos={y.sum()} ({y.mean():.2f})", flush=True)
    for seed in a.seeds:
        Xtr, Xte, ytr, yte = make_splits(X, y, test_size=0.30, seed=seed)
        imp = SimpleImputer(strategy="median").fit(Xtr)
        Xtr, Xte = imp.transform(Xtr), imp.transform(Xte)
        keep = Xtr.std(0) > 0
        Xtr, Xte = Xtr[:, keep], Xte[:, keep]
        sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        mk = dict(S.models(seed, len(ytr)))
        mk["FT-CUR (inter densa)"] = lambda: FTCURDense(random_state=seed, epochs=200, patience=200)
        row = {}
        for name, build in mk.items():
            if any(name.startswith(k) for k in a.skip) or (seed, name) in done:
                continue
            t0 = time.perf_counter(); torch.manual_seed(seed); np.random.seed(seed)
            clf = build().fit(Xtr, ytr)
            pred = clf.predict(Xte)
            try:
                auc = float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
            except Exception:
                auc = float("nan")
            S.M.SAINTStage.forward = S.ORIG_SAINT_FWD; S.W.FTTransformerClassifier = S.ORIG_FTC
            row[name] = float(f1_score(yte, pred, average="macro"))
            recs.append({"seed": seed, "model": name, "f1": row[name], "auc": auc,
                         "n_train": len(ytr), "p": int(Xtr.shape[1]),
                         "time_s": time.perf_counter() - t0})
            out.write_text(json.dumps(recs, indent=1))
        print(seed, "  ".join(f"{k} {v:.3f}" for k, v in row.items()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
