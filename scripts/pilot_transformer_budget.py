#!/usr/bin/env python3
"""Piloto do protocolo de treino dos Transformers ancorado nos artigos
(FT-Transformer: AdamW lr 1e-4, lote 256, paciência 16 épocas, sem teto;
 SAINT: AdamW lr 1e-4, lote 256, 100 épocas, melhor ponto de validação).

Compara, com UMA configuração fixa por modelo e S sementes, o protocolo publicado
(40 épocas, paciência 6, lr 1e-3, lotes 512/1024/completo) com o protocolo
proposto (lr 1e-4, lote 256, paciência 16 épocas, piso de MIN_EPOCHS, teto MAX).
Mede F1-macro, épocas efetivas e tempo — insumo para dimensionar a re-execução.

Uso: python scripts/pilot_transformer_budget.py --datasets TWS HAB AI4I BANK TELCO --seeds 5
     [--models FT_softmax SAINT FTCUR_full FTCUR_mb_global] [--protocols published paper]
"""
from __future__ import annotations
import argparse, json, sys, time, statistics as st
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from src.data.loaders import DatasetLoader

PROTOCOLS = {
    # (lr, batch FT, batch SAINT, batch FT-CUR, epochs, patience, min_epochs)
    "published": dict(lr=1e-3, bs_ft=512, bs_saint=1024, bs_ftcur=4096, epochs=40,  patience=6,  min_epochs=0),
    "paper":     dict(lr=1e-4, bs_ft=256, bs_saint=256,  bs_ftcur=256,  epochs=1000, patience=16, min_epochs=200),
}
TIER2 = {"ADULT", "BANK", "CREDIT", "HIGGS50K", "SHOPPERS", "TELCO"}

def make(model, P, seed):
    if model == "FT_softmax":
        from src.models.transformers.ft_transformer import FTTransformer
        return FTTransformer(embedding_dim=64, num_blocks=2, num_heads=2, dropout=0.1, lr=P["lr"],
                             batch_size=P["bs_ft"], max_epochs=P["epochs"], patience=P["patience"],
                             min_epochs=P["min_epochs"], random_state=seed)
    if model == "SAINT":
        from src.models.ft_transformer_saint_wrapper import SAINTColnorm
        return SAINTColnorm(d_model=32, n_heads=2, n_layers=1, lr=P["lr"], batch_size=P["bs_saint"],
                            epochs=P["epochs"], patience=P["patience"], early_stop_metric="val_loss",
                            min_epochs=P["min_epochs"], random_state=seed)
    from src.models.ft_transformer_cur_wrapper import FTTransformerCURColnorm
    kw = dict(d_model=32, n_heads=2, n_layers=1, m_ratio=0.2, lr=P["lr"], epochs=P["epochs"],
              patience=P["patience"], early_stop_metric="val_loss", min_epochs=P["min_epochs"],
              random_state=seed)
    if model == "FTCUR_full":
        return FTTransformerCURColnorm(batch_size=None, **kw)
    if model == "FTCUR_mb_global":
        return FTTransformerCURColnorm(batch_size=P["bs_ftcur"], minibatch_landmarks="global",
                                       predict_mode="streaming", **kw)
    if model == "FTCUR_mb_perbatch":
        return FTTransformerCURColnorm(batch_size=P["bs_ftcur"], minibatch_landmarks="per_batch", **kw)
    raise ValueError(model)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["TWS", "HAB", "AI4I", "BANK", "TELCO"])
    ap.add_argument("--models", nargs="+", default=["FT_softmax", "SAINT", "FTCUR_full", "FTCUR_mb_global"])
    ap.add_argument("--protocols", nargs="+", default=["published", "paper"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--output", default="results/pilot_transformer_budget.json")
    a = ap.parse_args()
    out = []
    for ds in a.datasets:
        X, y, _ = DatasetLoader.load(ds); y = (y > 0).astype(int) if set(np.unique(y)) == {-1, 1} else y.astype(int)
        for seed in range(a.seeds):
            if ds in TIER2:
                idx, _ = next(StratifiedShuffleSplit(1, train_size=2858, random_state=seed).split(X, y))
                Xs, ys = X[idx], y[idx]
            else:
                Xs, ys = X, y
            Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=0.30, stratify=ys, random_state=seed)
            sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            for prot in a.protocols:
                P = PROTOCOLS[prot]
                for model in a.models:
                    t0 = time.time()
                    try:
                        m = make(model, P, seed).fit(Xtr, ytr)
                        f1 = f1_score(yte, m.predict(Xte), average="macro")
                        n_ep = getattr(m, "n_iter_", None)
                        rec = dict(dataset=ds, seed=seed, protocol=prot, model=model, f1=round(float(f1), 4),
                                   n_epochs=n_ep, fit_s=round(time.time() - t0, 1), status="ok")
                    except Exception as e:
                        rec = dict(dataset=ds, seed=seed, protocol=prot, model=model, status="error", error=str(e)[:200])
                    out.append(rec); print(rec, flush=True)
                    Path(a.output).write_text(json.dumps(out, indent=1))
    ok = [r for r in out if r["status"] == "ok"]
    print("\n== resumo (F1 médio | tempo médio de ajuste) ==")
    for prot in a.protocols:
        for model in a.models:
            for ds in a.datasets:
                rs = [r for r in ok if r["protocol"] == prot and r["model"] == model and r["dataset"] == ds]
                if rs:
                    print(f"{prot:<10} {model:<18} {ds:<9} F1={st.mean(r['f1'] for r in rs):.3f}  fit={st.mean(r['fit_s'] for r in rs):6.1f}s  n={len(rs)}")

if __name__ == "__main__":
    main()
