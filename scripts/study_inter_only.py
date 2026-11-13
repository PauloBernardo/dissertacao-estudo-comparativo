#!/usr/bin/env python3
"""Atenção entre amostras SEM atenção entre atributos (sem rótulos no contexto).

Modelo: x → MLP de embedding (sem atenção entre atributos) → h → atenção entre amostras
(a camada do FT-CUR, com resíduo) → cabeça. Variantes:
  inter_cur   — atenção entre amostras comprimida (Nyström/CUR, m = 0,5 N, landmarks re-sorteados por época)
  inter_densa — atenção entre amostras densa
  sem_inter   — mesmo modelo sem a camada (um MLP)
Mede F1 e, nos modelos com atenção, se a predição depende do contexto: troca do contexto de treino
por reamostragem, colunas permutadas e ruído na predição.

    .venv/bin/python scripts/study_inter_only.py --seeds 0 1 2
"""
from __future__ import annotations

import argparse, copy, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT)]
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from src.data.loaders import DatasetLoader  # noqa: E402
from src.data.preprocessing import make_splits  # noqa: E402
from src.models.ft_transformer_model import InterInstanceAttentionNystrom  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402


def load(ds):
    if ds == "ARR":                               # Arrhythmia (UCI 5), mesma binarização do estudo
        raw = np.genfromtxt(ROOT / "data/raw/arrhythmia/arrhythmia.data", delimiter=",",
                            missing_values="?", filling_values=np.nan)
        return raw[:, :-1], (raw[:, -1].astype(int) != 1).astype(int)
    X, y, _ = DatasetLoader.load(ds); return X, np.asarray(y).astype(int)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InterOnly(nn.Module):
    def __init__(self, p, d=32, heads=4, use_inter=True):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(p, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d))
        self.inter = InterInstanceAttentionNystrom(d, heads) if use_inter else None
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d // 2), nn.ReLU(), nn.Linear(d // 2, 1))

    def forward(self, x, landmark_idx=None):
        h = self.emb(x)
        if self.inter is not None:
            h = self.inter(h, landmark_idx)
        return self.head(h).squeeze(-1)


def run(Xtr, ytr, Xte, kind, seed, m_ratio=0.5, epochs=1000, patience=150):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    T = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV)
    n = len(ytr); perm = rng.permutation(n); nv = max(1, round(0.2 * n)); iv, it = perm[:nv], perm[nv:]
    model = InterOnly(Xtr.shape[1], use_inter=(kind != "sem_inter")).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, yt, Xv, yv = T(Xtr[it]), T(ytr[it]), T(Xtr[iv]), T(ytr[iv]); nt = len(it)
    lm_of = (lambda k: torch.tensor(rng.choice(k, max(2, round(m_ratio * k)), replace=False), device=DEV)) \
        if kind == "inter_cur" else (lambda k: None)
    best, best_loss, bad = None, np.inf, 0
    for _ in range(epochs):
        model.train()
        loss = F.binary_cross_entropy_with_logits(model(Xt, lm_of(nt)), yt)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            vl = F.binary_cross_entropy_with_logits(model(torch.cat([Xt, Xv]), lm_of(nt))[nt:], yv).item()
        if vl < best_loss - 1e-4:
            best_loss, best, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best); model.eval()
    lm_te = lm_of(n)

    @torch.no_grad()
    def predict(X_ctx):                          # contexto (treino ou substituto) ‖ teste
        return model(torch.cat([T(X_ctx), T(Xte)]), lm_te)[len(X_ctx):].cpu().numpy()
    return predict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["TWC", "TWS", "PID", "VCP"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--output", default=str(ROOT / "results/study_inter_only.json"))
    a = ap.parse_args(); recs = []
    for ds in a.datasets:
        X, y = load(ds)
        for seed in a.seeds:
            Xtr, Xte, ytr, yte = make_splits(X, y, test_size=0.30, seed=seed)
            imp = SimpleImputer(strategy="median").fit(Xtr); Xtr, Xte = imp.transform(Xtr), imp.transform(Xte)
            keep = Xtr.std(0) > 0; Xtr, Xte = Xtr[:, keep], Xte[:, keep]
            for nome, clf in (("knn", KNeighborsClassifier(15)),
                              ("xgboost", XGBClassifier(n_estimators=200, max_depth=3, random_state=seed, verbosity=0))):
                Xs = StandardScaler().fit(Xtr)
                f = float(f1_score(yte, clf.fit(Xs.transform(Xtr), ytr).predict(Xs.transform(Xte)), average="macro"))
                recs.append({"ds": ds, "seed": seed, "model": nome, "f1": f}); print(f"{ds:4s} {seed} {nome:11s} F1 {f:.3f}", flush=True)
            sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            n, p = Xtr.shape; rng = np.random.default_rng(1000 + seed)
            ctxs = {"bootstrap": Xtr[rng.integers(0, n, n)],
                    "colperm": np.column_stack([Xtr[rng.permutation(n), j] for j in range(p)]),
                    "ruido": rng.standard_normal((n, p))}
            for kind in ("inter_cur", "inter_densa", "sem_inter"):
                pr = run(Xtr, ytr, Xte, kind, seed)
                pb = (pr(Xtr) > 0).astype(int); f0 = float(f1_score(yte, pb, average="macro"))
                rec = {"ds": ds, "seed": seed, "model": kind, "f1": f0}
                if kind != "sem_inter":
                    for c, Xc in ctxs.items():
                        pc = (pr(Xc) > 0).astype(int)
                        rec[c] = {"flip": float(np.mean(pc != pb)),
                                  "df1": float(f1_score(yte, pc, average="macro") - f0)}
                recs.append(rec); Path(a.output).write_text(json.dumps(recs, indent=1))
                extra = "  ".join(f"{c}: troca {rec[c]['flip']:.3f} ΔF1 {rec[c]['df1']:+.3f}"
                                  for c in ctxs if c in rec)
                print(f"{ds:4s} {seed} {kind:11s} F1 {f0:.3f}  {extra}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
