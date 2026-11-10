#!/usr/bin/env python3
"""Nyströmformer ENTRE ATRIBUTOS e LSSVM-ℓ1 (ADMM-Nyström) no Arrhythmia (452 × 279).

Motivação: nas bases do estudo (p ≤ 30) a atenção densa entre atributos é barata e o Nyström
só acrescenta erro. Com p = 279 (≈280 tokens por amostra) o custo O(p²) da softmax passa a
pesar — é o primeiro regime em que comprimir a atenção entre atributos pode se pagar.

Modelos (mesmo split/pré-processamento de study_intersample_arrhythmia.py):
  FT-Softmax          — FTTransformer, atenção densa entre atributos (O(p²) por amostra)
  FT-Nyström (m=k)    — mesmo modelo/hiperparâmetros; atenção Nyströmformer (Xiong et al.,
                        2021): landmarks = médias de segmentos de Q e K, pinv por Newton–Schulz
                        (6 iterações); nunca materializa a matriz p×p (O(p·m))
  ADMM-Nyström (ℓ1)   — LSSVM esparso; GridSearchCV 3-fold (σ ajustado à escala de p)
  XGBoost             — referência
Mede F1 macro, AUC, tempo, dispositivo e pico de memória da GPU (FT tenta a GPU; se der OOM,
cai para a CPU e registra).

    .venv/bin/python scripts/study_arrhythmia_feature_nystrom.py --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse, json, sys, time, types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402
from src.data.preprocessing import make_splits  # noqa: E402
from src.models.transformers.ft_transformer import FTTransformer, MultiHeadSparseAttention  # noqa: E402
from src.models.lssvm.primal.admm_nystrom import ADMMNystromLSSVM  # noqa: E402
from study_intersample_arrhythmia import load_arrhythmia  # noqa: E402


def _ns_pinv(W: torch.Tensor, n_iter: int = 6) -> torch.Tensor:
    """Pseudo-inversa iterativa do Nyströmformer (init de Xiong et al. 2021, Eq. 16)."""
    Z = W.transpose(-1, -2) / (W.abs().sum(-1).amax(-1, keepdim=True).unsqueeze(-1)
                               * W.abs().sum(-2).amax(-1, keepdim=True).unsqueeze(-1))
    I = torch.eye(W.shape[-1], device=W.device, dtype=W.dtype)
    for _ in range(n_iter):
        WZ = W @ Z
        Z = 0.25 * Z @ (13 * I - WZ @ (15 * I - WZ @ (7 * I - WZ)))
    return Z


def _nystrom_forward(self, x: torch.Tensor) -> torch.Tensor:
    B, L, D = x.shape
    H, HD, m = self.num_heads, self.head_dim, self.n_landmarks
    q = self.q_proj(x).view(B, L, H, HD).transpose(1, 2) * self.scale ** 0.5
    k = self.k_proj(x).view(B, L, H, HD).transpose(1, 2) * self.scale ** 0.5
    v = self.v_proj(x).view(B, L, H, HD).transpose(1, 2)
    if m >= L:                                   # sem compressão possível → densa
        w = F.softmax(q @ k.transpose(-2, -1), dim=-1)
        out = w @ v
    else:                                        # médias de segmentos (adaptive pooling)
        ql = F.adaptive_avg_pool1d(q.reshape(B * H, L, HD).transpose(1, 2), m).transpose(1, 2).view(B, H, m, HD)
        kl = F.adaptive_avg_pool1d(k.reshape(B * H, L, HD).transpose(1, 2), m).transpose(1, 2).view(B, H, m, HD)
        k1 = F.softmax(q @ kl.transpose(-2, -1), dim=-1)          # L×m
        k2 = F.softmax(ql @ kl.transpose(-2, -1), dim=-1)         # m×m
        k3 = F.softmax(ql @ k.transpose(-2, -1), dim=-1)          # m×L
        out = k1 @ (_ns_pinv(k2) @ (k3 @ v))
    out = out.transpose(1, 2).contiguous().view(B, L, D)
    return self.out_proj(out)


class FTNystrom(FTTransformer):
    """FTTransformer com atenção Nyströmformer entre os tokens de atributos."""

    def __init__(self, n_landmarks: int = 32, embedding_dim: int = 64, num_blocks: int = 3,
                 num_heads: int = 4, ffn_factor: int = 4, dropout: float = 0.1,
                 attn_dropout: float = 0.0, attention_type="softmax", topk_ratio: float = 0.25,
                 alpha: float = 1.5, lr: float = 1e-4, weight_decay: float = 1e-5,
                 batch_size: int = 256, max_epochs: int = 200, patience: int = 20,
                 val_fraction: float = 0.10, device=None, random_state: int = 42,
                 min_epochs: int = 0):
        super().__init__(embedding_dim, num_blocks, num_heads, ffn_factor, dropout, attn_dropout,
                         attention_type, topk_ratio, alpha, lr, weight_decay, batch_size,
                         max_epochs, patience, val_fraction, device, random_state, min_epochs)
        self.n_landmarks = n_landmarks

    def _build_model(self, n_features: int):
        net = super()._build_model(n_features)
        for mod in net.modules():
            if isinstance(mod, MultiHeadSparseAttention):
                mod.n_landmarks = self.n_landmarks
                mod.forward = types.MethodType(_nystrom_forward, mod)
        return net


def _fit_ft(build, Xtr, ytr):
    """Tenta a GPU; em OOM cai para a CPU. Retorna (clf, device, pico_MiB)."""
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            clf = build("cuda").fit(Xtr, ytr)
            return clf, "cuda", torch.cuda.max_memory_allocated() / 2**20
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
    return build("cpu").fit(Xtr, ytr), "cpu", float("nan")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    p.add_argument("--landmarks", nargs="+", type=int, default=[16, 32])
    p.add_argument("--models", nargs="+", default=["FT-Softmax", "FT-Nyström", "ADMM-Nyström (ℓ1)", "XGBoost"])
    p.add_argument("--output", default=str(ROOT / "results/study_arrhythmia_feature_nystrom.json"))
    a = p.parse_args()
    out = Path(a.output)
    recs = json.loads(out.read_text()) if out.exists() else []
    done = {(r["seed"], r["model"]) for r in recs}
    X, y = load_arrhythmia()
    print(f"Arrhythmia: {X.shape}", flush=True)
    for seed in a.seeds:
        Xtr, Xte, ytr, yte = make_splits(X, y, test_size=0.30, seed=seed)
        imp = SimpleImputer(strategy="median").fit(Xtr)
        Xtr, Xte = imp.transform(Xtr), imp.transform(Xte)
        keep = Xtr.std(0) > 0
        Xtr, Xte = Xtr[:, keep], Xte[:, keep]
        sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        pdim = Xtr.shape[1]
        ft = dict(random_state=seed, batch_size=256, max_epochs=200, patience=200)
        jobs = {}
        if "FT-Softmax" in a.models:
            jobs["FT-Softmax"] = ("ft", lambda dev: FTTransformer(attention_type="softmax", device=dev, **ft))
        if "FT-Nyström" in a.models:
            for m in a.landmarks:
                jobs[f"FT-Nyström (m={m})"] = ("ft", lambda dev, m=m: FTNystrom(n_landmarks=m, device=dev, **ft))
        if "ADMM-Nyström (ℓ1)" in a.models:
            s0 = np.sqrt(pdim)                    # distâncias típicas ~ √(2p) com dados padronizados
            grid = {"sigma": [0.5 * s0, s0, 2 * s0], "tau": [1.0, 0.1, 0.02],
                    "lambda_rel": [0.2, 0.05, 0.01, 0.001]}
            jobs["ADMM-Nyström (ℓ1)"] = ("cv", lambda: GridSearchCV(
                ADMMNystromLSSVM(m_ratio=0.30, rho="adaptive", max_iter=500, kernel_reg=True,
                                 landmark_method="random", random_state=seed),
                grid, scoring="f1_macro", cv=StratifiedKFold(3, shuffle=True, random_state=seed), n_jobs=4))
        if "XGBoost" in a.models:
            jobs["XGBoost"] = ("cv", lambda: XGBClassifier(n_estimators=200, max_depth=3, random_state=seed, verbosity=0))
        row = {}
        for name, (kind, build) in jobs.items():
            if (seed, name) in done:
                continue
            t0 = time.perf_counter(); torch.manual_seed(seed); np.random.seed(seed)
            yl = np.where(ytr == 1, 1, -1) if name.startswith("ADMM") else ytr
            if kind == "ft":
                clf, dev, peak = _fit_ft(build, Xtr, ytr)
            else:
                clf, dev, peak = build().fit(Xtr, yl), "cpu", float("nan")
            fit_s = time.perf_counter() - t0
            pred = np.asarray(clf.predict(Xte)); pred = (pred > 0).astype(int) if name.startswith("ADMM") else pred
            try:
                s = clf.predict_proba(Xte)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(Xte)
                auc = float(roc_auc_score(yte, s))
            except Exception:
                auc = float("nan")
            rec = {"seed": seed, "model": name, "f1": float(f1_score(yte, pred, average="macro")),
                   "auc": auc, "device": dev, "peak_gpu_mib": peak, "fit_s": fit_s, "p": pdim,
                   "n_train": len(ytr)}
            if name.startswith("ADMM"):
                best = clf.best_estimator_
                rec.update(best_params={k: float(v) for k, v in clf.best_params_.items()},
                           sparsity=float(best.sparsity_ratio_), n_active=int(best.n_support_))
            recs.append(rec); row[name] = rec["f1"]
            out.write_text(json.dumps(recs, indent=1))
            print(f"  {seed} {name:22s} F1 {rec['f1']:.3f} AUC {auc:.3f} {dev} "
                  f"{fit_s:6.0f}s pico {peak:7.0f} MiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
