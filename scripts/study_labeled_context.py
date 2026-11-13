#!/usr/bin/env python3
"""FT-CUR com CONTEXTO ROTULADO (inspirado em NPT/SPIN): a atenção entre amostras passa a valer
a pena se o contexto carregar rótulos?

Variante: um embedding do rótulo é somado ao [CLS] dos LANDMARKS (as chaves do contexto),
antes da atenção entre amostras. Regras contra vazamento:
  - só landmarks carregam rótulo (o bloco R agrega TODAS as chaves; se as demais carregassem
    rótulo, ele vazaria por ali);
  - a perda é calculada só nas amostras que NÃO são landmarks (ninguém vê o próprio rótulo);
    os landmarks são re-sorteados a cada época;
  - validação interna (20%) e teste nunca são landmarks: são consultas sem rótulo contra os
    landmarks rotulados do treino. Rótulos de teste não entram em momento algum.
Comparação no MESMO laço de treino (full-batch, Adam, melhor checkpoint de validação):
  FT-CUR rotulado / FT-CUR (CUR, sem rótulo) / FT (sem atenção entre amostras), + kNN de referência.
Teste de uso do contexto no modelo rotulado treinado: rótulos dos landmarks embaralhados
(e invertidos) na predição.

    .venv/bin/python scripts/study_labeled_context.py --seeds 0 1 2
"""
from __future__ import annotations

import argparse, copy, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT)]
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from src.data.loaders import DatasetLoader  # noqa: E402
from src.data.preprocessing import make_splits  # noqa: E402
from src.models.ft_transformer_model import FTTransformerClassifier  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LabeledCtxFT(FTTransformerClassifier):
    def __init__(self, n_features, use_labels: bool, **kw):
        super().__init__(n_features=n_features, attn_mode="nystrom", **kw)
        self.use_labels = use_labels
        self.label_emb = nn.Embedding(2, self.d_model)
        nn.init.normal_(self.label_emb.weight, std=0.1)

    def forward(self, x, landmark_idx=None, y_land=None):
        h = self.get_cls_embeddings(x)
        if self.use_inter_instance:
            if self.use_labels and y_land is not None:
                lab = torch.zeros_like(h)
                lab[landmark_idx] = self.label_emb(y_land)
                h = h + lab                       # rótulo SÓ nos landmarks
            h = self.inter_attn(h, landmark_idx)
        return self.head(h).squeeze(-1)


def fit_predict(Xtr, ytr, Xte, kind, seed, m_ratio=0.2, epochs=200, patience=30, lr=1e-3):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    n = len(ytr); perm = rng.permutation(n); nv = max(1, round(0.2 * n))
    iv, it = perm[:nv], perm[nv:]
    model = LabeledCtxFT(Xtr.shape[1], use_labels=(kind == "rotulado"), d_model=32, n_heads=4,
                         n_layers=2, use_inter_instance=(kind != "sem_inter")).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    T = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV)
    Xt, yt, Xv, yv = T(Xtr[it]), T(ytr[it]), T(Xtr[iv]), T(ytr[iv])
    nt = len(it); m = max(2, round(m_ratio * nt))
    best, best_loss, bad = None, np.inf, 0
    for ep in range(epochs):
        model.train()
        lm = torch.tensor(rng.choice(nt, m, replace=False), device=DEV)
        mask = torch.ones(nt, dtype=torch.bool, device=DEV); mask[lm] = False
        out = model(Xt, landmark_idx=lm if kind != "sem_inter" else None, y_land=yt[lm].long())
        loss = F.binary_cross_entropy_with_logits(out[mask], yt[mask])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():                    # validação: consultas sem rótulo
            ctx = torch.cat([Xt, Xv])
            lmv = torch.tensor(rng.choice(nt, m, replace=False), device=DEV)
            ov = model(ctx, landmark_idx=lmv if kind != "sem_inter" else None, y_land=yt[lmv].long())[nt:]
            vl = F.binary_cross_entropy_with_logits(ov, yv).item()
        if vl < best_loss - 1e-4:
            best_loss, best, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best); model.eval()
    Xall, yall = T(Xtr), torch.tensor(ytr, device=DEV).long()
    lm_te = torch.tensor(np.random.default_rng(seed + 7).choice(n, round(m_ratio * n), replace=False), device=DEV)

    @torch.no_grad()
    def predict(y_ctx=None, X_ctx=None):         # contexto = treino inteiro; teste sem rótulo
        yl = (yall if y_ctx is None else y_ctx)[lm_te]
        Xc = Xall if X_ctx is None else T(X_ctx)
        o = model(torch.cat([Xc, T(Xte)]), landmark_idx=lm_te if kind != "sem_inter" else None, y_land=yl)
        return o[n:].cpu().numpy()
    return predict, yall


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["BCW", "PID", "HAB", "VCP", "GCR", "AUS", "TWS", "TWM", "TWC"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--output", default=str(ROOT / "results/study_labeled_context.json"))
    a = ap.parse_args()
    out = Path(a.output); recs = json.loads(out.read_text()) if out.exists() else []
    done = {(r["ds"], r["seed"], r["model"]) for r in recs}
    for ds in a.datasets:
        X, y, _ = DatasetLoader.load(ds); y = np.asarray(y).astype(int)
        for seed in a.seeds:
            Xtr, Xte, ytr, yte = make_splits(X, y, test_size=0.30, seed=seed)
            sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            row = {}
            for kind in ("rotulado", "cur", "sem_inter", "knn"):
                if (ds, seed, kind) in done:
                    continue
                t0 = time.perf_counter(); rec = {"ds": ds, "seed": seed, "model": kind}
                if kind == "knn":
                    pred = KNeighborsClassifier(15).fit(Xtr, ytr).predict(Xte)
                    rec["f1"] = float(f1_score(yte, pred, average="macro"))
                else:
                    predict, yall = fit_predict(Xtr, ytr, Xte, kind, seed)
                    lg = predict(); pb = (lg > 0).astype(int)
                    rec["f1"] = float(f1_score(yte, pb, average="macro"))
                    if kind == "rotulado":       # o contexto rotulado é usado?
                        g = torch.Generator(device="cpu").manual_seed(seed)
                        for nome, yc in (("rot_embaralhado", yall[torch.randperm(len(yall), generator=g).to(yall.device)]),
                                         ("rot_invertido", 1 - yall)):
                            l2 = predict(yc); p2 = (l2 > 0).astype(int)
                            rec[nome] = {"flip": float(np.mean(p2 != pb)),
                                         "df1": float(f1_score(yte, p2, average="macro") - rec["f1"])}
                rec["s"] = time.perf_counter() - t0
                recs.append(rec); out.write_text(json.dumps(recs, indent=1)); row[kind] = rec
            if row:
                r = row.get("rotulado", {})
                extra = (f"  | embaralhado flip {r['rot_embaralhado']['flip']:.2f} ΔF1 {r['rot_embaralhado']['df1']:+.3f}"
                         f"  invertido flip {r['rot_invertido']['flip']:.2f} ΔF1 {r['rot_invertido']['df1']:+.3f}") if r else ""
                print(f"{ds:4s} {seed} " + "  ".join(f"{k} {v['f1']:.3f}" for k, v in row.items()) + extra, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
