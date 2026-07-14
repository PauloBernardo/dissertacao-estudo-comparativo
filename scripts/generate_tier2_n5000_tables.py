#!/usr/bin/env python3
"""Gera as tabelas LaTeX da ablação Tier 2 N=5000 (params fixos) para a tese.

Produz em ``results/tables/``:
    tier2_n5000_comparison.tex   — F1-macro N=2000 (GridCV) vs N=5000 (fixo), delta
    tier2_n5000_ftcur_m.tex      — FT-CUR: m (landmarks) por dataset, m = m_ratio*N
    tier2_n5000_admm_collapse.tex — ADMM-Nystrom por dataset, N=2000 vs N=5000

Uso:
    python scripts/generate_tier2_n5000_tables.py
"""

from __future__ import annotations

import collections
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"

DATASETS = ["ADULT", "BANK", "CREDIT", "HIGGS50K", "SHOPPERS", "TELCO"]

LABELS = {
    "StandardLSSVM": "LSSVM (Standard)", "DualFISTA": "LSSVM-DualFISTA",
    "PCPLSSVm": "LSSVM-PCP", "IPLSSVmOriginal": "LSSVM-IP",
    "OppositeMapsOriginalLSSVM": "LSSVM-OppMaps", "PruningLSSVM": "LSSVM-Pruning",
    "NystromLSSVMColnorm": "LSSVM-Nystrom", "ADMMNystromLSSVM": "ADMM-Nystrom",
    "FISTANystrom": "FISTA-Nystrom", "XGBoost": "XGBoost",
    "FTTransformer_softmax": "FT-Softmax", "FTTransformer_topk": "FT-TopK",
    "FTTransformer_entmax": "FT-Entmax", "FTTransformer_sparsemax": "FT-Sparsemax",
    "SAINTColnorm": "SAINT", "FTTransformerCURColnorm": "FT-CUR",
}


def _load_ok(*paths, n_train_filter=None):
    recs = []
    for p in paths:
        p = RESULTS / p
        if not p.exists():
            continue
        for r in json.loads(p.read_text()):
            if r.get("status") != "ok":
                continue
            if n_train_filter is not None and r.get("n_train_target", r.get("n_train")) != n_train_filter:
                continue
            recs.append(r)
    return recs


def main() -> None:
    # ── N=2000 (GridCV) — dedup transformers por (variant,dataset,seed) ────────
    recs2000 = [r for r in _load_ok("tier2_gridcv.json") if True]
    seen = set()
    for f in ["tier2_transformers.json", "tier2_transformers (1).json",
              "tier2_transformers (2).json", "tier2_transformers_merged.json"]:
        for r in _load_ok(f, n_train_filter=2000):
            variant = r.get("variant", r.get("model"))
            key = (variant, r.get("dataset"), r.get("seed"))
            if key in seen:
                continue
            seen.add(key)
            recs2000.append({**r, "variant": variant})

    by2000 = collections.defaultdict(list)
    for r in recs2000:
        by2000[r["variant"]].append(r["test_f1_macro"])

    # ── N=5000 (params fixos) ───────────────────────────────────────────────────
    recs5000 = _load_ok("tier2_fixedparams_n5000_lssvm.json",
                        "tier2_fixedparams_n5000_transformers.json")
    by5000 = collections.defaultdict(list)
    by5000_ds = collections.defaultdict(list)
    for r in recs5000:
        by5000[r["variant"]].append(r["test_f1_macro"])
        by5000_ds[(r["variant"], r["dataset"])].append(r["test_f1_macro"])

    # ── Tabela 1: comparação geral N=2000 vs N=5000 ─────────────────────────────
    rows = []
    for v in by5000:
        f5 = statistics.mean(by5000[v])
        f2 = statistics.mean(by2000[v]) if v in by2000 else None
        rows.append((v, f2, f5, (f5 - f2) if f2 is not None else None))
    rows.sort(key=lambda x: -x[2])

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{Comparação F1-macro entre $N_{\text{treino}}=2000$ (GridSearchCV) e "
        r"$N_{\text{treino}}=5000$ (hiperparâmetros fixos, moda do GridCV de $N=2000$), "
        r"30 sementes, 6 datasets.}",
        r"  \label{tab:tier2_n5000_comparison}",
        r"  \begin{tabular}{lrrr}",
        r"    \toprule",
        r"    Modelo & $F_1$ ($N{=}2000$) & $F_1$ ($N{=}5000$) & $\Delta$ \\",
        r"    \midrule",
    ]
    for v, f2, f5, delta in rows:
        label = LABELS.get(v, v)
        f2s = f"{f2:.4f}" if f2 is not None else "---"
        ds = f"{delta:+.4f}" if delta is not None else "---"
        lines.append(f"    {label} & {f2s} & {f5:.4f} & {ds} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    (TABLES / "tier2_n5000_comparison.tex").write_text("\n".join(lines))
    print("Escrito: tables/tier2_n5000_comparison.tex")

    # ── Tabela 2: FT-CUR m por dataset ──────────────────────────────────────────
    cur = [r for r in recs5000 if r["variant"] == "FTTransformerCURColnorm"]
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    m_ratio_ds = {}
    for r in cur:
        a = agg[r["dataset"]]
        a["m"].append(r.get("n_support_vectors"))
        a["sp"].append(r.get("sparsity_ratio"))
        a["nt"].append(r.get("n_train"))
        a["f1"].append(r["test_f1_macro"])
        m_ratio_ds.setdefault(r["dataset"], (r.get("fixed_params") or {}).get("m_ratio"))

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{FT-CUR em $N_{\text{treino}}=5000$: número de \textit{landmarks} "
        r"$m = m_{\text{ratio}} \cdot N$ por dataset (compressão constante, $m$ absoluto cresce com $N$).}",
        r"  \label{tab:tier2_n5000_ftcur_m}",
        r"  \begin{tabular}{lrrrrr}",
        r"    \toprule",
        r"    Dataset & $m_{\text{ratio}}$ & $N_{\text{treino}}$ & $m$ & Compressão & $F_1$ \\",
        r"    \midrule",
    ]
    for ds in DATASETS:
        if ds not in agg:
            continue
        a = agg[ds]
        mm = [x for x in a["m"] if x is not None]
        sp = [x for x in a["sp"] if x is not None]
        m_val = statistics.mean(mm) if mm else float("nan")
        sp_val = statistics.mean(sp) if sp else float("nan")
        lines.append(
            f"    {ds} & {m_ratio_ds.get(ds)} & {statistics.mean(a['nt']):.0f} & "
            f"{m_val:.0f} & {sp_val*100:.1f}\\% & {statistics.mean(a['f1']):.4f} \\\\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    (TABLES / "tier2_n5000_ftcur_m.tex").write_text("\n".join(lines))
    print("Escrito: tables/tier2_n5000_ftcur_m.tex")

    # ── Tabela 3: ADMM-Nystrom por dataset, N=2000 vs N=5000 ────────────────────
    adm2000 = collections.defaultdict(list)
    for r in recs2000:
        if r["variant"] == "ADMMNystromLSSVM":
            adm2000[r["dataset"]].append(r["test_f1_macro"])
    adm5000 = collections.defaultdict(list)
    for r in recs5000:
        if r["variant"] == "ADMMNystromLSSVM":
            adm5000[r["dataset"]].append(r["test_f1_macro"])

    order = sorted(DATASETS, key=lambda ds: statistics.mean(adm5000.get(ds, [float("nan")])))
    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{ADMM-Nystrom por dataset: $F_1$-macro em $N{=}2000$ (GridCV) vs.\ "
        r"$N{=}5000$ (hiperparâmetros $\ell_1$ fixos), ordenado pela maior queda.}",
        r"  \label{tab:tier2_n5000_admm_collapse}",
        r"  \begin{tabular}{lrrr}",
        r"    \toprule",
        r"    Dataset & $F_1$ ($N{=}2000$) & $F_1$ ($N{=}5000$) & $\Delta$ \\",
        r"    \midrule",
    ]
    for ds in order:
        f2 = statistics.mean(adm2000[ds]) if ds in adm2000 else float("nan")
        f5 = statistics.mean(adm5000[ds]) if ds in adm5000 else float("nan")
        lines.append(f"    {ds} & {f2:.4f} & {f5:.4f} & {f5-f2:+.4f} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    (TABLES / "tier2_n5000_admm_collapse.tex").write_text("\n".join(lines))
    print("Escrito: tables/tier2_n5000_admm_collapse.tex")

    # ── Tabela 4: validação por re-tuning (GridSearchCV real em N=5000) ─────────
    retune_path = RESULTS / "tier2_admm_regridcv_n5000.json"
    if retune_path.exists():
        recs_rt = [r for r in json.loads(retune_path.read_text()) if r.get("status") == "ok"]
        fixed_cfg = json.loads((ROOT / "config" / "tier2_fixed_params.json").read_text())

        by_rt = collections.defaultdict(list)
        params_rt = collections.defaultdict(list)
        for r in recs_rt:
            by_rt[r["dataset"]].append(r["test_f1_macro"])
            params_rt[r["dataset"]].append(r.get("best_params"))

        lines = [
            r"\begin{table}[ht]",
            r"  \centering",
            r"  \caption{Validação por re-tuning: ADMM-Nystrom em $N{=}5000$ com "
            r"\textit{GridSearchCV} completo (10 sementes), vs.\ $N{=}2000$ e vs.\ "
            r"$N{=}5000$ com hiperparâmetros fixos.}",
            r"  \label{tab:tier2_n5000_admm_retune}",
            r"  \begin{tabular}{lrrrr}",
            r"    \toprule",
            r"    Dataset & $F_1$ ($N{=}2000$) & $F_1$ ($N{=}5000$ fixo) & "
            r"$F_1$ ($N{=}5000$ retunado) & $n$ \\",
            r"    \midrule",
        ]
        for ds in ["CREDIT", "HIGGS50K"]:
            f2 = statistics.mean(adm2000[ds])
            f5 = statistics.mean(adm5000[ds])
            fr = statistics.mean(by_rt[ds])
            lines.append(f"    {ds} & {f2:.4f} & {f5:.4f} & \\textbf{{{fr:.4f}}} & {len(by_rt[ds])} \\\\")
        lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
        (TABLES / "tier2_n5000_admm_retune.tex").write_text("\n".join(lines))
        print("Escrito: tables/tier2_n5000_admm_retune.tex")

        for ds in ["CREDIT", "HIGGS50K"]:
            cnt = collections.Counter(
                tuple(sorted(p.items())) for p in params_rt[ds] if p)
            mode, n_mode = cnt.most_common(1)[0]
            print(f"  {ds}: fixo={fixed_cfg['ADMMNystromLSSVM'][ds]}  "
                  f"retunado(moda)={dict(mode)} ({n_mode}/{len(params_rt[ds])})")
    else:
        print("AVISO: tier2_admm_regridcv_n5000.json não encontrado — pulei a Tabela 4.")


if __name__ == "__main__":
    main()
