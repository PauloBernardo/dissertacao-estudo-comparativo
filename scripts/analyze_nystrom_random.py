#!/usr/bin/env python3
"""Análise: seleção de landmarks do Nyström-SVM — random vs colnorm.

Pareia os resultados por (dataset, seed) e testa, com Wilcoxon signed-rank,
se colnorm difere de random em F1-macro e acurácia. 30 seeds por dataset.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent

TIERS = {
    "tier1": {
        "colnorm":  ROOT / "results/tier1_gridcv.json",
        "random":   ROOT / "results/tier1_nystrom_random.json",
        "datasets": ["BCW", "PID", "HAB", "VCP", "GCR", "AUS", "AI4I", "TWS", "TWM", "TWC"],
    },
    "tier2": {
        "colnorm":  ROOT / "results/tier2_gridcv.json",
        "random":   ROOT / "results/tier2_nystrom_random.json",
        "datasets": ["ADULT", "BANK", "CREDIT", "HIGGS50K", "SHOPPERS", "TELCO"],
    },
}


def load(path, variant):
    recs = json.loads(Path(path).read_text())
    out = {}
    for r in recs:
        if r.get("variant") == variant and r.get("status") == "ok":
            out[(r["dataset"], r["seed"])] = r
    return out


def paired(col, rnd, ds, metric):
    seeds = sorted({s for (d, s) in col if d == ds} & {s for (d, s) in rnd if d == ds})
    c = [col[(ds, s)][metric] for s in seeds]
    r = [rnd[(ds, s)][metric] for s in seeds]
    return c, r, seeds


def wilcox(c, r):
    diffs = [a - b for a, b in zip(c, r)]
    if all(d == 0 for d in diffs):
        return float("nan"), 0
    try:
        stat, p = wilcoxon(c, r, zero_method="wilcox")
        return p, sum(1 for d in diffs if d != 0)
    except ValueError:
        return float("nan"), sum(1 for d in diffs if d != 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=list(TIERS), default="tier1")
    args = ap.parse_args()
    cfg = TIERS[args.tier]
    DATASETS = cfg["datasets"]
    tname = args.tier.replace("tier", "Tier ")

    col = load(cfg["colnorm"], "NystromLSSVMColnorm")
    rnd = load(cfg["random"], "NystromLSSVMRandom")

    for metric, label in [("test_f1_macro", "F1-macro"), ("test_accuracy", "Acurácia")]:
        print(f"\n{'='*78}\n{label}  —  colnorm vs random (Nyström-SVM, {tname}, 30 seeds)\n{'='*78}")
        print(f"{'dataset':8s} {'colnorm':>16s} {'random':>16s} {'Δ(col-rnd)':>11s} {'Wilcoxon p':>11s}")
        all_c, all_r = [], []
        wins_c = wins_r = ties = 0
        for ds in DATASETS:
            c, r, seeds = paired(col, rnd, ds, metric)
            if not seeds:
                print(f"{ds:8s}  (sem pares)")
                continue
            all_c += c; all_r += r
            mc, sc = st.mean(c), st.pstdev(c)
            mr, sr = st.mean(r), st.pstdev(r)
            d = mc - mr
            p, n_nz = wilcox(c, r)
            if d > 1e-6: wins_c += 1
            elif d < -1e-6: wins_r += 1
            else: ties += 1
            sig = "*" if (p == p and p < 0.05) else " "
            print(f"{ds:8s} {mc:7.4f}±{sc:5.3f} {mr:7.4f}±{sr:5.3f} "
                  f"{d:+11.4f} {p:11.4f}{sig} (n={len(seeds)})")

        mc, mr = st.mean(all_c), st.mean(all_r)
        p_all, _ = wilcox(all_c, all_r)
        print(f"{'-'*78}")
        print(f"{'GLOBAL':8s} {mc:7.4f}{'':9s} {mr:7.4f}{'':9s} {mc-mr:+11.4f} {p_all:11.4f}"
              f"  (n={len(all_c)} pares)")
        print(f"Datasets em que colnorm > random: {wins_c} | random > colnorm: {wins_r} | empate: {ties}")

    # Orçamento de landmarks / esparsidade escolhidos
    print(f"\n{'='*78}\nOrçamento de landmarks (sparsity_ratio médio) — efeito na comparação\n{'='*78}")
    print(f"{'dataset':8s} {'colnorm m/n':>12s} {'random m/n':>12s}")
    for ds in DATASETS:
        c, r, seeds = paired(col, rnd, ds, "sparsity_ratio")
        if not seeds:
            continue
        # m/n = 1 - sparsity
        print(f"{ds:8s} {1-st.mean(c):12.3f} {1-st.mean(r):12.3f}")


if __name__ == "__main__":
    main()
