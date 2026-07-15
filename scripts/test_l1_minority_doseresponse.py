#!/usr/bin/env python3
"""Dose-resposta: a poda ℓ1 da classe minoritária escala com o desbalanceamento?

Confirma (ou refuta) o mecanismo proposto para o COLAPSO PREDITIVO das formulações
primais ℓ1 DENSAS em dados reais desbalanceados (Tier 2):

    sob desbalanceamento, o termo de mínimos quadrados é dominado pelos resíduos da
    classe majoritária; os coeficientes ligados à minoria ficam de menor magnitude e
    são zerados PRIMEIRO pelo soft-threshold. Passado um ponto, a minoria não tem
    NENHUMA representação no modelo -> ele prediz só a majoritária -> F1-macro degenera.

MÉTRICA: razão de representação da minoria entre os coeficientes sobreviventes,

    r = (% minoria entre α != 0) / (% minoria no treino)

    r = 1  -> poda NEUTRA quanto à classe
    r < 1  -> minoria podada DESPROPORCIONALMENTE
    r = 0  -> minoria eliminada do modelo (colapso)

CONTROLE DE ESPARSIDADE: r é comparado a ESPARSIDADE EQUIPARADA (interpolado em
s* = 0,90), pois um modelo mais podado zera mais de tudo — sem isso, confundiríamos
"podou a minoria" com "podou muito".

DESENHO DOSE-RESPOSTA: os 6 datasets reais varrem um gradiente natural de
desbalanceamento (1,1:1 até 7,5:1). Se r(s*) DECRESCE com o desbalanceamento, o
desbalanceamento é a causa — e não uma coincidência de um dataset.

Os hiperparâmetros (sigma, tau) são escolhidos POR DATASET maximizando o F1 do LSSVM
denso SEM ℓ1: o teste precisa de um modelo que funcione antes de a penalidade entrar.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.data.loaders import DatasetLoader
from src.data.preprocessing import _convert_labels, make_splits
from src.models.lssvm.primal.admm_nesterov import ADMMNesterovLSSVM
from src.models.lssvm.standard import StandardLSSVM

ROOT = Path(__file__).resolve().parent.parent

# gradiente natural de desbalanceamento (razão majoritária:minoritária)
DATASETS = ["HIGGS50K", "COVER", "ADULT", "CREDIT", "SHOPPERS", "BANK"]
SEEDS = [0, 1, 2, 3, 4]
LAMBDAS = [0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.5]
SIGMAS, TAUS = [0.5, 2.0, 5.0, 10.0], [0.5, 5.0, 50.0]
NTRAIN = 2000
S_STAR = 0.90  # esparsidade de referência para a comparação equiparada

# Hiperparâmetros FIXOS por dataset. Isolar a seed = variar SÓ o sorteio dos dados;
# se (σ,τ) forem re-tunados a cada seed, a seed passa a trocar o próprio MODELO e a
# variância vira artefato do tuning, não do dado. Escolhidos por tuning pooled (seed 0)
# entre configs que atingem alta esparsidade — ver escolher_hparams_fixos() abaixo.
FIXED_HP: dict[str, tuple[float, float]] = {}  # preenchido em runtime


def prep(ds, seed):
    X, y, _ = DatasetLoader.load(ds)
    cap = round(NTRAIN / 0.70)
    if len(X) > cap:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=cap, random_state=seed)
        idx, _ = next(sss.split(X, y))
        X, y = X[idx], y[idx]
    Xtr, Xte, ytr, yte = make_splits(X, y, test_size=0.30, seed=seed)
    ytr = _convert_labels(ytr, "signed")
    yte = _convert_labels(yte, "signed")
    sc = StandardScaler()
    return sc.fit_transform(Xtr), sc.transform(Xte), ytr, yte


def tune_dense(Xtr, Xte, ytr, yte):
    """(sigma, tau) que maximiza o F1 do LSSVM denso SEM ℓ1."""
    best = (-1.0, None)
    for sg in SIGMAS:
        for tau in TAUS:
            m = StandardLSSVM(sigma=sg, tau=tau).fit(Xtr, ytr)
            f = f1_score(yte, m.predict(Xte), average="macro", zero_division=0)
            if f > best[0]:
                best = (f, (sg, tau))
    return best[1], best[0]


def escolher_hparams_fixos(ds):
    """(σ,τ) fixo p/ o dataset: melhor F1 denso na seed 0, exigindo que o modelo
    de fato ESPARSIFIQUE (senão r(s*) fica indefinido). Roda uma ADMM em λ alto para
    confirmar que a config alcança s* antes de aceitá-la."""
    Xtr, Xte, ytr, yte = prep(ds, seed=0)
    cand = []
    for sg in SIGMAS:
        for tau in TAUS:
            f = f1_score(yte, StandardLSSVM(sigma=sg, tau=tau).fit(Xtr, ytr).predict(Xte),
                         average="macro", zero_division=0)
            m = ADMMNesterovLSSVM(sigma=sg, tau=tau, lambda_=0.3, max_iter=500).fit(Xtr, ytr)
            s_hi = 1.0 - (np.abs(m.alpha_) > 1e-8).sum() / len(ytr)
            if s_hi >= S_STAR:                 # só aceita quem chega à esparsidade alvo
                cand.append((f, sg, tau))
    f, sg, tau = max(cand)
    return sg, tau


def interp_at(spars, ratios, s_star):
    """r interpolado na esparsidade s*; None se a curva não alcança s*."""
    pts = sorted((s, r) for s, r in zip(spars, ratios) if not np.isnan(r))
    if not pts or pts[-1][0] < s_star:
        return None
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    return float(np.interp(s_star, xs, ys))


def r_min_pre_colapso(spars, ratios, s_star):
    """Menor r observado ATÉ a esparsidade s* (inclusive, por interpolação).

    Leitura robusta: não depende de acertar um único ponto de esparsidade — captura o
    quanto a minoria chega a ser espremida no trajeto de esparsificação útil (s <= s*).
    """
    r_at = interp_at(spars, ratios, s_star)
    band = [r for s, r in zip(spars, ratios) if s <= s_star and not np.isnan(r)]
    if r_at is not None:
        band.append(r_at)
    return min(band) if band else None


def main():
    for ds in DATASETS:  # tuning fixo por dataset (pooled na seed 0), 1x
        FIXED_HP[ds] = escolher_hparams_fixos(ds)

    rows, per_ds = [], {}
    for ds in DATASETS:
        sg, tau = FIXED_HP[ds]                       # <<< FIXO para todas as sementes
        r_star_list, r_min_list, imb = [], [], None
        for seed in SEEDS:
            Xtr, Xte, ytr, yte = prep(ds, seed)
            vals, cnts = np.unique(ytr, return_counts=True)
            minor = vals[np.argmin(cnts)]
            p_orig = float((ytr == minor).mean())
            imb = (1 - p_orig) / p_orig
            f1_dense = f1_score(yte, StandardLSSVM(sigma=sg, tau=tau).fit(Xtr, ytr)
                                .predict(Xte), average="macro", zero_division=0)

            spars, ratios = [], []
            collapsed_at = None
            for lam in LAMBDAS:
                m = ADMMNesterovLSSVM(sigma=sg, tau=tau, lambda_=lam, max_iter=500)
                m.fit(Xtr, ytr)
                nz = np.abs(m.alpha_) > 1e-8
                n_surv = int(nz.sum())
                if n_surv == 0:
                    continue
                spars.append(1.0 - n_surv / len(ytr))
                ratios.append(float((ytr[nz] == minor).mean()) / p_orig)
                if collapsed_at is None and (m.predict(Xte) == minor).mean() == 0.0:
                    collapsed_at = lam

            r_star = interp_at(spars, ratios, S_STAR)
            r_min = r_min_pre_colapso(spars, ratios, S_STAR)
            if r_star is not None:
                r_star_list.append(r_star)
            if r_min is not None:
                r_min_list.append(r_min)
            rows.append(dict(dataset=ds, seed=seed, imbalance=imb, sigma=sg, tau=tau,
                             f1_dense=f1_dense, r_at_s90=r_star, r_min_pre=r_min,
                             collapse_lambda=collapsed_at, spars=spars, ratios=ratios))
            per_ds.setdefault(ds, []).append((r_star, r_min, collapsed_at))

        print(f"{ds:10s} razão {imb:4.1f}:1  (σ={sg},τ={tau})  "
              f"r(s=0,90)={np.mean(r_star_list):.3f}±{np.std(r_star_list):.3f}  "
              f"r_min={np.mean(r_min_list):.3f}±{np.std(r_min_list):.3f}  "
              f"(n={len(r_star_list)})")

    # ── dose-resposta (usa r_min: leitura robusta) ───────────────────────────
    print(f"\n{'='*90}\nDOSE-RESPOSTA: r_min (menor representação da minoria até s=0,90) "
          f"vs desbalanceamento\n{'='*90}")
    print(f"{'dataset':10s} {'razão':>8s} {'r_min méd':>11s} {'±dp':>7s} "
          f"{'colapsos':>9s} {'interpretação':>22s}")
    xs_all, ys_all = [], []
    for ds in DATASETS:
        rmins = [rm for _, rm, _ in per_ds[ds] if rm is not None]
        ncol = sum(1 for _, _, c in per_ds[ds] if c is not None)
        if not rmins:
            continue
        imb = [r["imbalance"] for r in rows if r["dataset"] == ds][0]
        r_m, r_s = float(np.mean(rmins)), float(np.std(rmins))
        for v in rmins:                               # ponto por seed p/ Spearman
            xs_all.append(imb); ys_all.append(v)
        tag = ("minoria ELIMINADA" if r_m < 0.15 else
               "minoria podada" if r_m < 0.70 else "poda ~neutra")
        print(f"{ds:10s} {imb:6.1f}:1 {r_m:11.3f} {r_s:7.3f} "
              f"{ncol:>3d}/{len(per_ds[ds]):<3d} {tag:>22s}")

    if len(set(xs_all)) >= 3:
        rho, p = spearmanr(xs_all, ys_all)
        print(f"\nSpearman(desbalanceamento, r_min) = {rho:+.3f}  (p = {p:.4f}, "
              f"n={len(xs_all)} pontos por seed)")
        print("Mecanismo confirmado se rho NEGATIVO e significativo "
              "(mais desbalanceado -> minoria mais espremida).")

    out = ROOT / "results" / "l1_minority_doseresponse.json"
    out.write_text(json.dumps(rows, indent=2, default=float))
    print(f"\nsalvo em {out}")


if __name__ == "__main__":
    main()
