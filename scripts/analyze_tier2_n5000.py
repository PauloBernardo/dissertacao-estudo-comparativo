#!/usr/bin/env python3
"""Análise da ablação Tier 2 em N=5000 (params fixos, 30 seeds).

Mescla os dois JSONs (LSSVMs+XGBoost rodados na CPU + Transformers no Kaggle),
e produz:
  1. Ranking F1-macro (média ± dp) + rank médio de Friedman.
  2. Teste de Friedman + Wilcoxon pareado (vs. melhor modelo, correção de Holm).
  3. Tabela do FT-CUR: m absoluto (landmarks) por dataset + compressão + F1,
     documentando que m = m_ratio * N (cresce com N), não m fixo.
  4. Comparação N=2000 -> N=5000 (ressalva: N=2000 usou GridCV; N=5000, params fixos).

Uso:
    python scripts/analyze_tier2_n5000.py \
        --lssvm results/tier2_fixedparams_n5000_lssvm.json \
        --transformers results/tier2_fixedparams_n5000_transformers.json
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics.statistical import friedman_test, average_ranks, wilcoxon_pairwise

DATASETS = ["ADULT", "BANK", "CREDIT", "HIGGS50K", "SHOPPERS", "TELCO"]

# Nome interno -> rótulo da tese
LABELS = {
    "StandardLSSVM": "LSSVM (Standard)", "DualFISTA": "LSSVM-DualFISTA",
    "PCPLSSVm": "LSSVM-PCP", "FSALSSVm": "LSSVM-FSA", "IPLSSVm": "LSSVM-IP",
    "NystromLSSVMColnorm": "LSSVM-Nystrom", "ADMMNystromLSSVM": "ADMM-Nystrom",
    "FISTANystrom": "FISTA-Nystrom", "XGBoost": "XGBoost",
    "FTTransformer_softmax": "FT-Softmax", "FTTransformer_topk": "FT-TopK",
    "FTTransformer_entmax": "FT-Entmax", "FTTransformer_sparsemax": "FT-Sparsemax",
    "SAINTColnorm": "SAINT", "FTTransformerCURColnorm": "FT-CUR",
}


def _load_ok(*paths: Path) -> list[dict]:
    recs = []
    for p in paths:
        if p.exists():
            recs += [r for r in json.loads(p.read_text()) if r.get("status") == "ok"]
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lssvm", type=Path,
                    default=ROOT / "results/tier2_fixedparams_n5000_lssvm.json")
    ap.add_argument("--transformers", type=Path,
                    default=ROOT / "results/tier2_fixedparams_n5000_transformers.json")
    ap.add_argument("--metric", default="test_f1_macro")
    args = ap.parse_args()

    recs = _load_ok(args.lssvm, args.transformers)
    if not recs:
        raise SystemExit("Nenhum registro OK encontrado.")

    m = args.metric
    # F1 por modelo (todas seeds/datasets) e média por (modelo, dataset)
    by_model = collections.defaultdict(list)
    by_md    = collections.defaultdict(list)
    for r in recs:
        v = r["variant"]
        by_model[v].append(r[m])
        by_md[(v, r["dataset"])].append(r[m])

    models = sorted(by_model, key=lambda v: -statistics.mean(by_model[v]))

    # ── 1. Ranking + rank de Friedman ──────────────────────────────────────────
    # matriz (datasets x modelos) de F1 médio por dataset (para ranks/Friedman)
    complete = [v for v in models
                if all((v, ds) in by_md for ds in DATASETS)]
    score_mat = np.array([[statistics.mean(by_md[(v, ds)]) for v in complete]
                          for ds in DATASETS])
    ranks = average_ranks(score_mat)
    rank_of = {v: ranks[j] for j, v in enumerate(complete)}

    print(f"\n{'='*64}\n  ABLAÇÃO TIER 2 — N=5000, params fixos, {len(recs)} registros OK\n{'='*64}")
    print(f"\n{'Modelo':<20}{'F1-macro (méd±dp)':>20}{'rank':>7}{'n':>6}")
    print("-" * 53)
    for v in models:
        fs = by_model[v]
        rk = f"{rank_of[v]:.2f}" if v in rank_of else "  --"
        print(f"{LABELS.get(v, v):<20}{statistics.mean(fs):>12.4f}±{statistics.pstdev(fs):.3f}"
              f"{rk:>7}{len(fs):>6}")

    # ── 2. Friedman + Wilcoxon (vs. melhor), Holm ──────────────────────────────
    if len(complete) >= 3:
        fr = friedman_test(score_mat)
        print(f"\nFriedman (n={len(DATASETS)} datasets, k={len(complete)} modelos): "
              f"stat={fr['statistic']:.3f}, p={fr['pvalue']:.4g}")

        best = complete[int(np.argmin(ranks))]
        jbest = complete.index(best)
        pvals = []
        for j, v in enumerate(complete):
            if v == best:
                continue
            res = wilcoxon_pairwise(score_mat[:, jbest], score_mat[:, j])
            pvals.append((v, res.get("pvalue", float("nan"))))
        # Holm
        pvals.sort(key=lambda x: (np.nan_to_num(x[1], nan=1.0)))
        k = len(pvals)
        print(f"\nWilcoxon vs. {LABELS.get(best, best)} (melhor rank), Holm-corrigido:")
        for i, (v, p) in enumerate(pvals):
            p_holm = min(1.0, p * (k - i)) if p == p else float("nan")
            sig = "*" if (p_holm == p_holm and p_holm < 0.05) else " "
            print(f"  {LABELS.get(v, v):<20} p={p:.4g}  p_holm={p_holm:.4g} {sig}")

    # ── 3. FT-CUR: m absoluto por dataset (m = m_ratio * N, CRESCE com N) ───────
    cur = [r for r in recs if r["variant"] == "FTTransformerCURColnorm"]
    if cur:
        print(f"\n{'-'*53}\nFT-CUR — m (landmarks) por dataset  [m = m_ratio · N_train]")
        print(f"{'dataset':<10}{'m_ratio':>8}{'n_train':>9}{'m':>7}{'compr.':>9}{'F1':>9}")
        seen = {}
        agg = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in cur:
            agg[r["dataset"]]["m"].append(r.get("n_support_vectors"))
            agg[r["dataset"]]["sp"].append(r.get("sparsity_ratio"))
            agg[r["dataset"]]["nt"].append(r.get("n_train"))
            agg[r["dataset"]]["f1"].append(r[m])
            seen.setdefault(r["dataset"], (r.get("fixed_params") or {}).get("m_ratio"))
        for ds in DATASETS:
            if ds not in agg:
                continue
            a = agg[ds]
            mm = [x for x in a["m"] if x is not None]
            sp = [x for x in a["sp"] if x is not None]
            print(f"{ds:<10}{str(seen.get(ds)):>8}{statistics.mean(a['nt']):>9.0f}"
                  f"{(statistics.mean(mm) if mm else float('nan')):>7.0f}"
                  f"{(statistics.mean(sp) if sp else float('nan')):>8.1%}"
                  f"{statistics.mean(a['f1']):>9.4f}")
        print("Nota: m cresce com N (m_ratio fixo) -> custo O(m_ratio·N²), NÃO o "
              "O(N·m) linear\n(que exige m fixo; ver benchmark de scaling).")


if __name__ == "__main__":
    main()
