#!/usr/bin/env python3
"""FT-CUR: o ganho com N é limitado pelo LOTE, não pelos landmarks?

No protocolo principal o FT-CUR usa batch_size=4096, maior que o conjunto de
ajuste em todos os regimes, logo treina em lote completo: 40 passos de gradiente
sempre, independentemente de N. O SAINT (lote 1024) passa de 2 para 4 passos por
época ao ir de N=2000 para N=5000, e é o modelo com maior ganho ao escalar.

Este estudo isola o fator: mesmo modelo, mesmos hiperparâmetros, mudando só o
lote (e, com lote menor, usando landmarks GLOBAIS + predição streaming, para o
contexto continuar o do conjunto inteiro).

    full     : batch_size=None  → 1 passo/época   (protocolo publicado)
    mb1024   : batch_size=1024 + landmarks globais → 2 (N=2000) / 4 (N=5000)

Uso: python scripts/study_ftcur_batch_confound.py [--datasets HIGGS50K CREDIT BANK] [--seeds 5]
"""
from __future__ import annotations
import argparse, json, sys, time, statistics as st, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from src.data.loaders import DatasetLoader
from src.experiments.reproducibility import set_global_seed
from src.models.ft_transformer_cur_wrapper import FTTransformerCURColnorm

FIXED = dict(d_model=32, lr=1e-3, epochs=40, patience=6, early_stop_metric='val_loss')

def run(ds, n_train, seed, mode, params):
    X, y, _ = DatasetLoader.load(ds)
    y = (y > 0).astype(int) if set(np.unique(y)) == {-1, 1} else y.astype(int)
    cap = int(round(n_train / 0.7))
    if len(X) > cap:
        idx, _ = next(StratifiedShuffleSplit(1, train_size=cap, random_state=seed).split(X, y))
        X, y = X[idx], y[idx]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)
    sc = StandardScaler().fit(Xtr)
    kw = dict(FIXED, **params)
    if mode == 'full':
        kw.update(batch_size=None)
    else:
        kw.update(batch_size=1024, minibatch_landmarks='global', predict_mode='streaming')
    set_global_seed(seed)
    t0 = time.time()
    m = FTTransformerCURColnorm(**kw, random_state=seed).fit(sc.transform(Xtr), ytr)
    f1 = f1_score(yte, m.predict(sc.transform(Xte)), average='macro')
    return dict(dataset=ds, n_train=n_train, seed=seed, mode=mode, f1=round(float(f1), 4),
                epochs=getattr(m, 'n_epochs_', None), fit_s=round(time.time() - t0, 1), params=params)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['HIGGS50K', 'CREDIT', 'BANK'])
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--output', default='results/ftcur_batch_confound.json')
    a = ap.parse_args()
    cfg = json.load(open('config/tier2_fixed_params.json'))['FTTransformerCURColnorm']
    out = []
    for ds in a.datasets:
        params = {k: v for k, v in cfg[ds].items()}
        for n_train in (2000, 5000):
            for mode in ('full', 'mb1024'):
                for seed in range(a.seeds):
                    try:
                        r = run(ds, n_train, seed, mode, params)
                    except Exception as e:
                        r = dict(dataset=ds, n_train=n_train, seed=seed, mode=mode, error=str(e)[:200])
                    out.append(r); print(r, flush=True)
                    Path(a.output).write_text(json.dumps(out, indent=1))
    ok = [r for r in out if 'f1' in r]
    print(f"\n{'ds':<10}{'modo':>8}{'N=2000':>9}{'N=5000':>9}{'Δ':>9}")
    for ds in a.datasets:
        for mode in ('full', 'mb1024'):
            g = {n: [r['f1'] for r in ok if r['dataset'] == ds and r['mode'] == mode and r['n_train'] == n] for n in (2000, 5000)}
            if g[2000] and g[5000]:
                print(f"{ds:<10}{mode:>8}{st.mean(g[2000]):>9.4f}{st.mean(g[5000]):>9.4f}{st.mean(g[5000])-st.mean(g[2000]):>+9.4f}")

if __name__ == '__main__':
    main()
