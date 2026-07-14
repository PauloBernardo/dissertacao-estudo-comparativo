#!/usr/bin/env python3
"""Ablação D — random vs colnorm no Nyström-SVM em N=5000 (params fixos).

Pareia por (dataset, seed) os resultados de N=5000 com hiperparâmetros fixos
(moda do GridCV de N=2000) e testa Wilcoxon signed-rank. Também reporta o
delta N=2000 → N=5000 de cada método (escalabilidade amostral).
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
COL_5000 = ROOT / "results/tier2_fixedparams_n5000_lssvm.json"
RND_5000 = ROOT / "results/tier2_fixedparams_n5000_nystrom_random.json"
COL_2000 = ROOT / "results/tier2_gridcv.json"
RND_2000 = ROOT / "results/tier2_nystrom_random.json"
DATASETS = ["ADULT", "BANK", "CREDIT", "HIGGS50K", "SHOPPERS", "TELCO"]


def load(path, variant, n5000=False):
    out = {}
    for r in json.loads(Path(path).read_text()):
        if r.get("variant") != variant or r.get("status") != "ok":
            continue
        if n5000 and r.get("n_train_target") != 5000:
            continue
        out[(r["dataset"], r["seed"])] = r
    return out


def wilcox(c, r):
    if all(a == b for a, b in zip(c, r)):
        return float("nan")
    try:
        return wilcoxon(c, r, zero_method="wilcox")[1]
    except ValueError:
        return float("nan")


def main():
    col = load(COL_5000, "NystromLSSVMColnorm", n5000=True)
    rnd = load(RND_5000, "NystromLSSVMRandom", n5000=True)
    col2 = load(COL_2000, "NystromLSSVMColnorm")
    rnd2 = load(RND_2000, "NystromLSSVMRandom")

    for metric, label in [("test_f1_macro", "F1-macro"), ("test_accuracy", "Acurácia")]:
        print(f"\n{'='*82}\n{label}  —  colnorm vs random  (Ablação D: Nyström-SVM, N=5000, params fixos)\n{'='*82}")
        print(f"{'dataset':9s} {'colnorm':>16s} {'random':>16s} {'Δ(col-rnd)':>11s} {'Wilcoxon p':>11s}")
        ac, ar = [], []
        wc = wr = 0
        for ds in DATASETS:
            seeds = sorted({s for (d, s) in col if d == ds} & {s for (d, s) in rnd if d == ds})
            if not seeds:
                print(f"{ds:9s}  (sem pares)"); continue
            c = [col[(ds, s)][metric] for s in seeds]
            r = [rnd[(ds, s)][metric] for s in seeds]
            ac += c; ar += r
            d = st.mean(c) - st.mean(r)
            p = wilcox(c, r)
            wc += d > 1e-6; wr += d < -1e-6
            sig = "*" if (p == p and p < 0.05) else " "
            print(f"{ds:9s} {st.mean(c):7.4f}±{st.pstdev(c):5.3f} {st.mean(r):7.4f}±{st.pstdev(r):5.3f} "
                  f"{d:+11.4f} {p:11.4f}{sig}")
        p_all = wilcox(ac, ar)
        print("-" * 82)
        print(f"{'GLOBAL':9s} {st.mean(ac):7.4f}{'':9s} {st.mean(ar):7.4f}{'':9s} "
              f"{st.mean(ac)-st.mean(ar):+11.4f} {p_all:11.4f}  (n={len(ac)})")
        print(f"colnorm > random: {wc} | random > colnorm: {wr} datasets")

    # Escalabilidade amostral: delta N=2000 → N=5000 por método
    print(f"\n{'='*82}\nEscalabilidade amostral  (F1-macro médio: N=2000 → N=5000)\n{'='*82}")
    print(f"{'dataset':9s} {'colnorm 2k→5k':>22s} {'random 2k→5k':>22s}")
    for ds in DATASETS:
        def mean_at(store, ds):
            v = [r["test_f1_macro"] for (d, s), r in store.items() if d == ds]
            return st.mean(v) if v else float("nan")
        c2, c5 = mean_at(col2, ds), mean_at(col, ds)
        r2, r5 = mean_at(rnd2, ds), mean_at(rnd, ds)
        print(f"{ds:9s}  {c2:.4f} → {c5:.4f} ({c5-c2:+.4f})   {r2:.4f} → {r5:.4f} ({r5-r2:+.4f})")
    # global
    def gmean(store):
        return st.mean([r["test_f1_macro"] for r in store.values()])
    print("-" * 82)
    print(f"{'GLOBAL':9s}  {gmean(col2):.4f} → {gmean(col):.4f} ({gmean(col)-gmean(col2):+.4f})"
          f"   {gmean(rnd2):.4f} → {gmean(rnd):.4f} ({gmean(rnd)-gmean(rnd2):+.4f})")


if __name__ == "__main__":
    main()
