#!/usr/bin/env python3
"""FT-CUR com atenção entre instâncias DENSA (sem CUR) — mesmo backbone, mesmo treino.

Responde: o FT-CUR ganharia como o SAINT no tabuleiro (TWC) se a atenção entre instâncias
não fosse comprimida? Usa o modo "atenção completa" de InterInstanceAttentionNystrom
(landmark_idx=None → softmax(QKᵀ)V sobre todas as amostras). Treino em lote único (N pequeno)
e predição transdutiva sobre [X_train ‖ X_test], o mesmo contrato do SAINT (o teste atende
ao treino e a si mesmo). Mesmo split/subamostragem/semente de study_intersample_scarce.py,
de modo que os números se pareiam com study_intersample_scarce_n{N}.json.

    .venv/bin/python scripts/study_ftcur_dense_inter.py --n-train 280 --datasets TWC
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from src.data.loaders import DatasetLoader  # noqa: E402
from src.data.preprocessing import make_splits  # noqa: E402
from src.models.ft_transformer_cur_wrapper import FTTransformerCURColnorm, DEVICE  # noqa: E402
from src.models.ft_transformer_model import FTTransformerClassifier, fit_model  # noqa: E402


class FTCURDense(FTTransformerCURColnorm):
    """Mesmo wrapper/hiperparâmetros do FT-CUR, mas atenção entre instâncias densa."""

    def fit(self, X, y):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        rng = np.random.RandomState(self.random_state)
        n = X.shape[0]; n_val = max(1, round(self.val_fraction * n))
        idx = rng.permutation(n); iv, it = idx[:n_val], idx[n_val:]
        self._model = FTTransformerClassifier(
            n_features=X.shape[1], d_model=self.d_model, n_heads=self.n_heads,
            n_layers=self.n_layers, use_inter_instance=True, tau_ratio=self.tau_ratio,
            attn_mode="nystrom", pinv_grad=self.pinv_grad).to(DEVICE)
        Xt, yt = self._to_tensor(X[it], y[it]); Xv, yv = self._to_tensor(X[iv], y[iv])
        fit_model(self._model, Xt, yt, Xv, yv, landmark_idx=None, lr=self.lr,
                  epochs=self.epochs, patience=self.patience, weight_decay=self.weight_decay,
                  early_stop_metric=self.early_stop_metric, batch_size=None,
                  global_landmark_rows=None, min_epochs=self.min_epochs)
        self.X_train_ = X
        return self

    @torch.no_grad()
    def _logits(self, X):
        self._model.eval()
        ctx = np.concatenate([self.X_train_, X], axis=0)
        out = self._model(self._to_tensor(ctx), landmark_idx=None)
        return out[len(self.X_train_):].cpu().numpy()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=280)
    p.add_argument("--datasets", nargs="+", default=["TWC"])
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    a = p.parse_args()
    out = ROOT / f"results/study_ftcur_dense_inter_n{a.n_train}.json"
    recs = []
    for ds in a.datasets:
        X, y, _ = DatasetLoader.load(ds); y = np.asarray(y).astype(int)
        for seed in a.seeds:
            Xtr, Xte, ytr, yte = make_splits(X, y, test_size=0.30, seed=seed)
            if a.n_train < len(ytr):
                Xtr, _, ytr, _ = train_test_split(Xtr, ytr, train_size=a.n_train, stratify=ytr,
                                                  random_state=seed)
            sc = StandardScaler().fit(Xtr)
            t0 = time.perf_counter(); torch.manual_seed(seed); np.random.seed(seed)
            clf = FTCURDense(random_state=seed, epochs=200, patience=200).fit(sc.transform(Xtr), ytr)
            f1 = float(f1_score(yte, clf.predict(sc.transform(Xte)), average="macro"))
            recs.append({"ds": ds, "seed": seed, "model": "FT-CUR (inter densa)", "f1": f1,
                         "n_train": len(ytr), "time_s": time.perf_counter() - t0})
            out.write_text(json.dumps(recs, indent=1))
            print(ds, seed, f"{f1:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
