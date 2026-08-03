#!/usr/bin/env python3
"""Como o limiar efetivo do ADMM-Nyström escala com N.

O passo de encolhimento do ADMM usa ``threshold = λ₁/(2ρ)``, e ρ não é um
hiperparâmetro livre: o modelo o fixa em ``ρ = 1/λ_max(CᵀC/τ)``. Logo

    threshold = λ₁ · λ_max(CᵀC/τ) / 2,

isto é, o encolhimento efetivo é proporcional ao maior autovalor do Gram
reduzido — que cresce com N, porque CᵀC acumula uma parcela por amostra.
Transferir λ₁ de N=2000 para N=5000 **sem** reajuste transporta o rótulo do
hiperparâmetro, não sua ação: o mesmo λ₁ passa a encolher muito mais.

Este script mede λ_max, ρ e o limiar nos dois valores de N, sob o mesmo
carregamento, subamostragem, partição e hiperparâmetros modais usados pela
Ablação D, de modo que os números sejam diretamente comparáveis aos dela.

Uso:
    python scripts/study_admm_rho_scaling.py [--seeds 3]
Saída:
    results/admm_rho_scaling.json
    dissertacao-latex/tables/admm_rho_scaling.tex
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_tier2_gridcv as g  # noqa: E402  (mesmo módulo usado pelo runner oficial)

THESIS = ROOT.parent / "dissertacao-latex"
VARIANT = "ADMMNystromLSSVM"
DATASETS = ["ADULT", "BANK", "CREDIT", "HIGGS50K", "SHOPPERS", "TELCO"]
NS = [2000, 5000]


def medir(dataset: str, seed: int, n_train: int, config: dict) -> dict:
    """Reproduz o preparo da Ablação D e devolve λ_max, ρ e o limiar."""
    cfg = g.GRIDS[VARIANT]
    g.set_global_seed(seed)

    X_full, y_full, _ = g.DatasetLoader.load(dataset)
    X_sub, y_sub = g._subsample(X_full, y_full, round(n_train / 0.70), seed)
    X_train, _, y_train_raw, _ = g.make_splits(X_sub, y_sub, test_size=0.30, seed=seed)
    y_train = g._convert_labels(y_train_raw, g._label_format(VARIANT))

    pipeline, _ = g._build_pipeline(VARIANT, seed)
    modal = config[VARIANT][dataset]
    pipeline.set_params(**{f"clf__{k}": v for k, v in modal.items()})
    pipeline.fit(X_train, y_train)

    clf = pipeline.named_steps["clf"]
    rho = float(clf.rho_used_)
    lam = float(clf.lambda_)
    return {"dataset": dataset, "seed": seed, "n_train": int(len(X_train)),
            "lambda": lam, "rho": rho, "lambda_max": 1.0 / rho,
            "threshold": lam / (2.0 * rho), "modal": modal}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()

    config = json.loads((ROOT / "config" / "tier2_fixed_params.json").read_text())
    regs = []
    for ds in DATASETS:
        for n in NS:
            for seed in range(a.seeds):
                try:
                    r = medir(ds, seed, n, config)
                except Exception as exc:                      # dataset menor que o alvo
                    print(f"  {ds:9s} N={n} seed={seed}: {type(exc).__name__}: {exc}")
                    continue
                regs.append(r)
                print(f"  {ds:9s} N={n} seed={seed}: lambda_max={r['lambda_max']:9.1f} "
                      f"rho={r['rho']:.3e} limiar={r['threshold']:.4f}")
    (ROOT / "results" / "admm_rho_scaling.json").write_text(json.dumps(regs, indent=2))

    linhas, razoes = [], []
    for ds in DATASETS:
        col = {}
        for n in NS:
            v = [r for r in regs if r["dataset"] == ds and r["n_train"] ==
                 max({x["n_train"] for x in regs if x["dataset"] == ds and
                      abs(x["n_train"] - n * 0.999) < n * 0.5}, default=-1)]
            v = [r for r in regs if r["dataset"] == ds and abs(r["n_train"] - n) < n * 0.15]
            if v:
                col[n] = (float(np.mean([x["lambda_max"] for x in v])),
                          float(np.mean([x["threshold"] for x in v])),
                          float(np.mean([x["lambda"] for x in v])))
        if len(col) != 2:
            continue
        lm2, th2, lam = col[2000]
        lm5, th5, _ = col[5000]
        razoes.append(lm5 / lm2)
        linhas.append(rf"    {ds} & {lam:.4f} & {lm2:.1f} & {lm5:.1f} & "
                      rf"{lm5 / lm2:.2f}$\times$ & {th2:.4f} & {th5:.4f} \\"
                      .replace(".", "{,}").replace("{,}1f", ".1f"))
    tex = "\n".join([
        r"\begin{table}[H]", r"  \centering",
        r"  \caption[Escalonamento do limiar efetivo do ADMM-Nyström com $N$]{Escalonamento do limiar efetivo do ADMM-Nyström com $N$. "
        r"Como $\rho = 1/\lambda_{\max}(\mathbf{C}^\top\mathbf{C}/\tau)$, o encolhimento "
        r"aplicado a cada iteração vale $\lambda_1\lambda_{\max}/2$ e cresce com o maior "
        r"autovalor do Gram reduzido. Transferir $\lambda_1$ de $N=2000$ para $N=5000$ "
        r"preserva o valor nominal do hiperparâmetro, mas multiplica sua ação pela razão "
        r"da penúltima coluna. Médias sobre 3 sementes, mesmos hiperparâmetros modais e "
        r"mesmo preparo de dados da Ablação~D.}",
        r"  \label{tab:admm_rho_scaling}",
        r"  \begin{tabular}{lrrrrrr}", r"    \toprule",
        r"    & & \multicolumn{3}{c}{$\lambda_{\max}(\mathbf{C}^\top\mathbf{C}/\tau)$} "
        r"& \multicolumn{2}{c}{limiar $\lambda_1/(2\rho)$} \\",
        r"    \cmidrule(lr){3-5}\cmidrule(lr){6-7}",
        r"    \textit{Dataset} & $\lambda_1$ & $N{=}2000$ & $N{=}5000$ & razão & "
        r"$N{=}2000$ & $N{=}5000$ \\",
        r"    \midrule", *linhas, r"    \bottomrule",
        r"  \end{tabular}", r"\end{table}", ""])
    (THESIS / "tables" / "admm_rho_scaling.tex").write_text(tex, encoding="utf-8")
    if razoes:
        print(f"\nrazão lambda_max (N=5000 / N=2000): média {np.mean(razoes):.2f}x, "
              f"min {min(razoes):.2f}x, max {max(razoes):.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
