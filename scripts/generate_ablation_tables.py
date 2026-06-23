#!/usr/bin/env python3
"""Generate LaTeX tables for ablation studies A, B, C.

Ablação A — Escalabilidade (N=400 → N=2000):
    Compares Tier 1 (TWS/TWM/TWC, N≈280 train) vs ablation (TWS_2k/TWM_2k/TWC_2k, N≈1400 train).

Ablação B — Ruído (2D → 5D):
    Compares Tier 1 (TWS/TWM/TWC, 2 features) vs ablation (TWS_5f/TWM_5f/TWC_5f, 5 features).

Ablação C — Multifeature sintético MK5:
    Absolute F1 on MKE/MKM/MKH (5-feature multikernel datasets); compared against Tier 1 avg.

Outputs (in results/tables/):
    ablation_a.tex
    ablation_b.tex
    ablation_c.tex

Usage
-----
    python scripts/generate_ablation_tables.py [--tier1 results/tier1_gridcv.json]
                                               [--abl-a results/ablation_a_scaling.json]
                                               [--abl-b results/ablation_b_noise.json]
                                               [--abl-c results/ablation_c_mk5.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TABLES_DIR = ROOT / "results" / "tables"

MODEL_ORDER = [
    "NystromLSSVMColnorm",
    "StandardLSSVM",
    "DualFISTA",
    "IPLSSVm",
    "FSALSSVm",
    "PCPLSSVm",
    "XGBoost",
    "PruningLSSVM",
    "FISTANesterov",
    "ADMMNesterovLSSVM",
    "ADMMElasticNet",
    "OppositeMapsLSSVM",
]

MODEL_LABELS: dict[str, str] = {
    "StandardLSSVM":       "LSSVM (Std)",
    "PCPLSSVm":            "LSSVM-PCP",
    "FSALSSVm":            "LSSVM-FSA",
    "PruningLSSVM":        "LSSVM-Pruning",
    "IPLSSVm":             "LSSVM-IP",
    "OppositeMapsLSSVM":   "LSSVM-OppMaps",
    "ADMMNesterovLSSVM":   "LSSVM-ADMM",
    "ADMMElasticNet":      "LSSVM-ADMM-EN",
    "FISTANesterov":       "LSSVM-FISTA",
    "DualFISTA":           "LSSVM-DualFISTA",
    "NystromLSSVMColnorm": "LSSVM-Nyström",
    "XGBoost":             "XGBoost",
}


def load_json(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text())
    df = pd.DataFrame(raw)
    df = df[df["status"] == "ok"].copy()
    df["model"] = df["variant"].fillna(df.get("model", pd.Series(dtype=str)))
    return df


def mean_f1(df: pd.DataFrame, model: str, dataset: str) -> float | None:
    sub = df[(df["model"] == model) & (df["dataset"] == dataset)]["test_f1_macro"]
    return float(sub.mean()) if len(sub) > 0 else None


def fmt(v: float | None, bold: bool = False) -> str:
    if v is None:
        return "---"
    s = f"{v:.3f}"
    return f"\\textbf{{{s}}}" if bold else s


def fmt_delta(delta: float | None) -> str:
    if delta is None:
        return "---"
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:+.3f}"


# ── Ablação A: Escalabilidade ──────────────────────────────────────────────────

def table_ablation_a(tier1: pd.DataFrame, abl: pd.DataFrame) -> str:
    pairs = [("TWS", "TWS_2k"), ("TWM", "TWM_2k"), ("TWC", "TWC_2k")]
    ds_labels = {"TWS": "TWS", "TWM": "TWM", "TWC": "TWC"}

    rows = []
    for variant in MODEL_ORDER:
        label = MODEL_LABELS.get(variant, variant)
        cols = [label]
        deltas = []
        for ds_t1, ds_2k in pairs:
            v1 = mean_f1(tier1, variant, ds_t1)
            v2 = mean_f1(abl,   variant, ds_2k)
            d  = (v2 - v1) if (v1 is not None and v2 is not None) else None
            cols += [fmt(v1), fmt(v2), fmt_delta(d)]
            if d is not None:
                deltas.append(d)
        avg_d = np.mean(deltas) if deltas else None
        cols.append(fmt_delta(avg_d))
        rows.append(cols)

    col_spec = "l" + "ccc" * 3 + "c"
    header1  = (
        r"\multicolumn{1}{c}{} & "
        r"\multicolumn{3}{c}{TWS} & "
        r"\multicolumn{3}{c}{TWM} & "
        r"\multicolumn{3}{c}{TWC} & "
        r"\multicolumn{1}{c}{} \\"
    )
    header2 = r"Modelo & $N_{400}$ & $N_{2k}$ & $\Delta$ & $N_{400}$ & $N_{2k}$ & $\Delta$ & $N_{400}$ & $N_{2k}$ & $\Delta$ & $\overline{\Delta}$ \\"

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{Ablação A — Escalabilidade amostral: F1-macro médio (20 sementes) em N=400 vs N=2000.}",
        r"  \label{tab:ablation_a}",
        r"  \footnotesize",
        rf"  \begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        f"    {header1}",
        r"    \cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        f"    {header2}",
        r"    \midrule",
    ]
    for row in rows:
        lines.append("    " + " & ".join(row) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ── Ablação B: Ruído (5D) ──────────────────────────────────────────────────────

def table_ablation_b(tier1: pd.DataFrame, abl: pd.DataFrame) -> str:
    pairs = [("TWS", "TWS_5f"), ("TWM", "TWM_5f"), ("TWC", "TWC_5f")]

    rows = []
    for variant in MODEL_ORDER:
        label = MODEL_LABELS.get(variant, variant)
        cols = [label]
        deltas = []
        for ds_2d, ds_5d in pairs:
            v1 = mean_f1(tier1, variant, ds_2d)
            v2 = mean_f1(abl,   variant, ds_5d)
            d  = (v2 - v1) if (v1 is not None and v2 is not None) else None
            cols += [fmt(v1), fmt(v2), fmt_delta(d)]
            if d is not None:
                deltas.append(d)
        avg_d = np.mean(deltas) if deltas else None
        cols.append(fmt_delta(avg_d))
        rows.append(cols)

    col_spec = "l" + "ccc" * 3 + "c"
    header1 = (
        r"\multicolumn{1}{c}{} & "
        r"\multicolumn{3}{c}{TWS} & "
        r"\multicolumn{3}{c}{TWM} & "
        r"\multicolumn{3}{c}{TWC} & "
        r"\multicolumn{1}{c}{} \\"
    )
    header2 = r"Modelo & 2D & 5D & $\Delta$ & 2D & 5D & $\Delta$ & 2D & 5D & $\Delta$ & $\overline{\Delta}$ \\"

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{Ablação B — Robustez ao ruído: F1-macro médio (20 sementes) com 2 vs 5 features.}",
        r"  \label{tab:ablation_b}",
        r"  \footnotesize",
        rf"  \begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        f"    {header1}",
        r"    \cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        f"    {header2}",
        r"    \midrule",
    ]
    for row in rows:
        lines.append("    " + " & ".join(row) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ── Ablação C: MK5 ────────────────────────────────────────────────────────────

def table_ablation_c(abl: pd.DataFrame) -> str:
    datasets = ["MKE", "MKM", "MKH"]

    rows = []
    for variant in MODEL_ORDER:
        label = MODEL_LABELS.get(variant, variant)
        cols = [label]
        vals = []
        for ds in datasets:
            v = mean_f1(abl, variant, ds)
            cols.append(fmt(v))
            if v is not None:
                vals.append(v)
        avg = np.mean(vals) if vals else None
        cols.append(fmt(avg))
        rows.append(cols)

    col_spec = "lcccc"
    header = r"Modelo & MKE & MKM & MKH & Média \\"

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{Ablação C — Síntético multifeature (MK5): F1-macro médio (20 sementes).}",
        r"  \label{tab:ablation_c}",
        r"  \footnotesize",
        rf"  \begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        f"    {header}",
        r"    \midrule",
    ]
    for row in rows:
        lines.append("    " + " & ".join(row) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tier1", default="results/tier1_gridcv.json")
    p.add_argument("--abl-a", default="results/ablation_a_scaling.json")
    p.add_argument("--abl-b", default="results/ablation_b_noise.json")
    p.add_argument("--abl-c", default="results/ablation_c_mk5.json")
    args = p.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    tier1 = load_json(Path(args.tier1))
    print(f"Tier 1: {len(tier1)} runs, {tier1['model'].nunique()} models, {tier1['dataset'].nunique()} datasets")

    for abl_key, path_attr, table_fn, out_name in [
        ("A", "abl_a", lambda t, a: table_ablation_a(t, a), "ablation_a.tex"),
        ("B", "abl_b", lambda t, a: table_ablation_b(t, a), "ablation_b.tex"),
        ("C", "abl_c", lambda t, a: table_ablation_c(a),    "ablation_c.tex"),
    ]:
        path = Path(getattr(args, path_attr.replace("-", "_")))
        if not path.exists():
            print(f"Ablação {abl_key}: {path} not found — skipping")
            continue
        abl = load_json(path)
        ok  = len(abl)
        print(f"Ablação {abl_key}: {ok} runs, {abl['dataset'].nunique()} datasets")
        tex = table_fn(tier1, abl)
        out = TABLES_DIR / out_name
        out.write_text(tex)
        print(f"  Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
