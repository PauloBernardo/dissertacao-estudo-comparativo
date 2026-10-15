"""FT-CUR: inferência indutiva (resumo congelado) contra a transdutiva publicada (2026-10-15).

Na Ablação D (N = 5000, params fixos, ajuste final 200 épocas sem paciência), o FT-CUR prediz em
predict_mode="full_context": concatena [X_train ‖ X_test] e recalcula o resumo de Nyström com os
pontos de teste dentro. Aqui, o MESMO modelo treinado prediz também em "streaming": K_m e
pinv(W)·R·V congelados só com o treino (O(m·d)), sem guardar X_train.

Para cada (base, semente) registra:
  f1_full          F1 em full_context (deve reproduzir o publicado)
  f1_stream        F1 em streaming
  agree            fração de predições iguais entre os dois modos
  stream_1by1_eq   streaming uma amostra por vez == streaming em lote (indutivo de verdade)
  full_1by1_agree  full_context uma amostra por vez vs em lote (dependência do lote de teste), 200 pts
  summary_kb       tamanho do resumo congelado (K_m + resumo, float32)

Uso: python scripts/study_ftcur_streaming_inference.py [--seeds 0 1 2] [--datasets ...]
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.metrics import f1_score
import refit_tf_nopatience as RF   # aplica o orçamento 200/200 do protocolo publicado
R1, R2 = RF.R1, RF.R2
from src.experiments.reproducibility import set_global_seed

SRC = Path("results/tier2_fixedparams_n5000_transformers.json")
OUT = Path("results/ftcur_streaming_inference.json")


def run(src: dict) -> dict:
    variant, dataset, seed = src["variant"], src["dataset"], src["seed"]
    set_global_seed(seed)
    X, y, _ = R1.DatasetLoader.load(dataset)
    X, y = R2._subsample(X, y, round(5000 / 0.70), seed)
    Xtr, Xte, ytr, yte = R1.make_splits(X, y, test_size=0.30, seed=seed)
    fmt = R2._label_format(variant)
    ytr, yte = R1._convert_labels(ytr, fmt), R1._convert_labels(yte, fmt)
    pipe, _ = R2._build_pipeline(variant, seed)
    pipe.set_params(**{f"clf__{k}": v for k, v in src["fixed_params"].items()})
    pipe.fit(Xtr, ytr)
    clf = pipe.named_steps["clf"]

    clf.predict_mode = "full_context"
    y_full = pipe.predict(Xte)
    k = min(200, len(Xte))
    y_full_1 = np.array([pipe.predict(Xte[i:i + 1])[0] for i in range(k)])

    clf._precompute_train_summary()
    clf.predict_mode = "streaming"
    y_str = pipe.predict(Xte)
    y_str_1 = np.array([pipe.predict(Xte[i:i + 1])[0] for i in range(k)])

    kb = (clf._stream_K_m.numel() + clf._stream_summary.numel()) * 4 / 1024
    return dict(dataset=dataset, seed=seed, params=src["fixed_params"], m=int(clf.n_landmarks_),
                f1_published=src["test_f1_macro"],
                f1_full=f1_score(yte, y_full, average="macro"), f1_stream=f1_score(yte, y_str, average="macro"),
                agree=float(np.mean(y_full == y_str)),
                stream_1by1_eq=bool(np.array_equal(y_str_1, y_str[:k])),
                full_1by1_agree=float(np.mean(y_full_1 == y_full[:k])),
                summary_kb=round(kb, 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--datasets", nargs="+", default=None)
    a = ap.parse_args()
    todo = [r for r in json.loads(SRC.read_text())
            if r["variant"] == "FTTransformerCURRandom" and r.get("status") == "ok" and r["seed"] in a.seeds
            and (a.datasets is None or r["dataset"] in a.datasets)]
    out = []
    for src in sorted(todo, key=lambda r: (r["dataset"], r["seed"])):
        rec = run(src); out.append(rec)
        OUT.write_text(json.dumps(out, indent=1))
        print(f"{rec['dataset']:8s} s{rec['seed']}  pub {rec['f1_published']:.4f}  full {rec['f1_full']:.4f}  "
              f"stream {rec['f1_stream']:.4f}  concord {rec['agree']:.3f}  stream1x1={rec['stream_1by1_eq']}  "
              f"full1x1 {rec['full_1by1_agree']:.3f}  m={rec['m']}  resumo {rec['summary_kb']} KB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
