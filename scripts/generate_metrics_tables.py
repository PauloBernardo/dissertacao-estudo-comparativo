#!/usr/bin/env python3
"""Tabelas de métricas complementares (acurácia, MCC, AUC-ROC) por tier.

A métrica primária do estudo é o F1-macro, adequada ao desbalanceamento dos
conjuntos. Este script gera as tabelas de apoio que permitem verificar se a
hierarquia de desempenho se sustenta sob métricas alternativas, e calcula a
correlação de Spearman entre os *ranks* de cada métrica e os do F1-macro.

Saídas:
    dissertacao-latex/tables/tier{1,2}_metrics.tex
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
THESIS = ROOT.parent / "dissertacao-latex"

KEY = {
    'ADMMNesterovLSSVM': 'LSSVM-ADMM', 'ADMMElasticNet': 'LSSVM-ADMM-Elastic',
    'ADMMNystromLSSVM': 'ADMM-Nystrom', 'DualFISTA': 'LSSVM-DualFISTA',
    'FISTANesterov': 'LSSVM-FISTA', 'FISTANystrom': 'FISTA-Nystrom',
    'FSALSSVmOriginal': 'LSSVM-FSA', 'IPLSSVmOriginal': 'LSSVM-IP',
    'NystromLSSVMColnorm': 'LSSVM-Nystrom', 'OppositeMapsOriginalLSSVM': 'LSSVM-OppMaps',
    'PCPLSSVm': 'LSSVM-PCP', 'PruningLSSVM': 'LSSVM-Pruning',
    'StandardLSSVM': 'LSSVM (Standard)', 'XGBoost': 'XGBoost', 'SAINTColnorm': 'SAINT',
    'FTTransformerCURColnorm': 'FT-CUR', 'FTTransformer_softmax': 'FT-Softmax',
    'FTTransformer_topk': 'FT-TopK', 'FTTransformer_entmax': 'FT-Entmax',
    'FTTransformer_sparsemax': 'FT-Sparsemax',
}
METRICS = [
    ('test_f1_macro', 'F1-macro'),
    ('test_accuracy', 'Acurácia'),
    ('test_mcc',      'MCC'),
    ('test_auc_roc',  'AUC-ROC'),
]

TIERS = {
    1: (['results/tier1_gridcv.json'],
        ['AI4I', 'AUS', 'BCW', 'GCR', 'HAB', 'PID', 'TWC', 'TWM', 'TWS', 'VCP']),
    2: (['results/tier2_gridcv.json', 'results/tier2_transformers.json'],
        ['ADULT', 'BANK', 'CREDIT', 'HIGGS50K', 'SHOPPERS', 'TELCO']),
}


def collect(files: list[str]) -> dict:
    d: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for f in files:
        for r in json.load(open(ROOT / f)):
            k = KEY.get(r.get('variant')) or KEY.get(r.get('model'))
            if not k or r.get('status') != 'ok':
                continue
            for met, _ in METRICS:
                if r.get(met) is not None:
                    d[met][k][r['dataset']].append(r[met])
    return d


def build(tier: int) -> str:
    files, ds = TIERS[tier]
    d = collect(files)
    models = sorted(d['test_f1_macro'])

    means, ranks = {}, {}
    for met, _ in METRICS:
        M = np.array([[np.mean(d[met][m][x]) if d[met][m].get(x) else np.nan
                       for x in ds] for m in models])
        means[met] = np.nanmean(M, axis=1)
        ranks[met] = np.array([stats.rankdata(-M[:, j]) for j in range(len(ds))]).mean(axis=0)

    order = np.argsort(ranks['test_f1_macro'])
    body = []
    for i in order:
        cells = " & ".join(
            f"{means[met][i]:.3f} ({ranks[met][i]:.1f})" for met, _ in METRICS)
        body.append(f"    {models[i]} & {cells} \\\\")

    rho = {met: stats.spearmanr(ranks['test_f1_macro'], ranks[met])[0]
           for met, _ in METRICS[1:]}
    rho_txt = ", ".join(
        f"{lbl}: $\\rho = {rho[met]:.3f}$".replace('.', '{,}')
        for met, lbl in METRICS[1:])

    return "\n".join([
        r"\begin{table}[H]",
        r"  \centering",
        rf"  \caption[Métricas complementares --- Tier {tier}]{{Métricas complementares --- Tier {tier}. Cada célula traz a "
        rf"média sobre os {len(ds)} \textit{{datasets}} e 30 sementes, com o "
        rf"\textit{{rank}} médio de Friedman entre parênteses (menor = melhor). "
        rf"Ordenado pelo \textit{{rank}} de F1-macro. Correlação de Spearman entre "
        rf"os \textit{{ranks}} de cada métrica e os de F1-macro --- {rho_txt}.}}",
        rf"  \label{{tab:tier{tier}_metrics}}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{lcccc}",
        r"    \toprule",
        r"    Modelo & " + " & ".join(lbl for _, lbl in METRICS) + r" \\",
        r"    \midrule",
        *body,
        r"    \bottomrule",
        r"  \end{tabular}}",
        r"\end{table}",
        "",
    ])


def main() -> int:
    for tier in (1, 2):
        out = THESIS / "tables" / f"tier{tier}_metrics.tex"
        out.write_text(build(tier), encoding="utf-8")
        print(f"  -> {out.relative_to(THESIS.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
