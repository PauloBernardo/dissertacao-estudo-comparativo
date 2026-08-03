#!/usr/bin/env python3
"""Validação externa: concordância entre este estudo e o TabZilla.

O TabZilla \\cite{mcelfresh2023when} avalia 19 algoritmos sobre 176 conjuntos
de dados e publica os resultados brutos (`metadataset_clean.csv`, ~363 MB, em
https://drive.google.com/drive/folders/1cHisTmruPHDCYVOYnaqvTdybLngMkB8R).
Cinco dos conjuntos do Tier 1 deste estudo coincidem exatamente com conjuntos
do TabZilla, o que permite confrontar, modelo a modelo, valores obtidos por
duas implementações e dois protocolos independentes.

    HAB -> openml__haberman__42          PID -> openml__diabetes__37
    BCW -> openml__wdbc__9946            GCR -> openml__credit-g__31
    AUS -> openml__Australian__146818

Métrica: AUC-ROC de teste. O F1 do TabZilla usa `average="weighted"` (e, para
tarefas binárias, `average="micro"`, que coincide com a acurácia), ao passo que
a métrica primária deste estudo é o F1-macro — os dois não são comparáveis.
AUC-ROC e acurácia são definidos identicamente nos dois estudos.

Protocolo de cada lado:
    TabZilla : partição 80/10/10, 10 folds, 30 configurações de hiperparâmetros
               (1 default + 29 aleatórias); escolhe-se aqui a configuração de
               melhor AUC de *validação* e reporta-se seu AUC de *teste*.
    Este     : 30 sementes, GridSearchCV no treino, AUC de teste médio.

Uso:
    python scripts/compare_tabzilla.py --metadataset /caminho/metadataset_clean.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
THESIS = ROOT.parent / "dissertacao-latex"

TZ_DS = {"openml__haberman__42": "HAB", "openml__diabetes__37": "PID",
         "openml__wdbc__9946": "BCW", "openml__credit-g__31": "GCR",
         "openml__Australian__146818": "AUS"}
DS = ["AUS", "BCW", "GCR", "HAB", "PID"]

# Modelos homólogos: (rótulo nesta tese, alg_name no TabZilla, rótulo na tabela)
PARES = [("XGBoost",    "XGBoost",            "XGBoost"),
         ("FT-Softmax", "rtdl_FTTransformer", "FT-Transformer"),
         ("SAINT",      "SAINT",              "SAINT")]
# Modelos sem homólogo exato: o representante kernel do TabZilla é uma SVC cujo
# único hiperparâmetro ajustado é C (gamma fica no 'scale' do sklearn).
KERNEL = [("LSSVM-Std", "LSSVM (Standard)"), ("Nystrom-SVM", "Nyström-SVM")]

KEY = {"StandardLSSVM": "LSSVM-Std", "NystromLSSVMColnorm": "Nystrom-SVM",
       "XGBoost": "XGBoost", "SAINTColnorm": "SAINT",
       "FTTransformer_softmax": "FT-Softmax"}


def carrega_nosso() -> dict:
    d: dict = defaultdict(lambda: defaultdict(list))
    for r in json.loads((ROOT / "results" / "tier1_gridcv.json").read_text()):
        k = KEY.get(r.get("variant")) or KEY.get(r.get("model"))
        if k and r.get("status") == "ok" and r["dataset"] in DS \
                and r.get("test_auc_roc") is not None:
            d[k][r["dataset"]].append(r["test_auc_roc"])
    return {m: {ds: float(np.mean(v)) for ds, v in dd.items()} for m, dd in d.items()}


def carrega_tabzilla(path: Path) -> dict:
    cols = ["dataset_name", "alg_name", "hparam_source", "AUC__val", "AUC__test"]
    chunks = [c[c.dataset_name.isin(TZ_DS)]
              for c in pd.read_csv(path, usecols=cols, chunksize=200_000)]
    d = pd.concat(chunks)
    d["ds"] = d.dataset_name.map(TZ_DS)
    g = (d.groupby(["ds", "alg_name", "hparam_source"])
           .agg(val=("AUC__val", "mean"), test=("AUC__test", "mean")).reset_index())
    # configuração escolhida pela validação, nunca pelo teste
    best = g.sort_values("val", ascending=False).groupby(["ds", "alg_name"]).head(1)
    return {a: dict(zip(gr.ds, gr.test)) for a, gr in best.groupby("alg_name")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadataset", required=True, type=Path)
    a = ap.parse_args()

    O, T = carrega_nosso(), carrega_tabzilla(a.metadataset)
    linhas, todos_o, todos_t = [], [], []

    print("=== Concordância em modelos homólogos (AUC-ROC de teste) ===")
    for nosso, deles, rotulo in PARES:
        dd = [d for d in DS if d in T.get(deles, {}) and not np.isnan(T[deles][d])]
        o = np.array([O[nosso][d] for d in dd])
        t = np.array([T[deles][d] for d in dd])
        mae = float(np.mean(np.abs(o - t)))
        print(f"\n{rotulo:15s} ({len(dd)} bases)")
        for d, x, y in zip(dd, o, t):
            print(f"   {d:4s} este={x:.4f}  TabZilla={y:.4f}  |dif|={abs(x - y):.4f}")
        print(f"   MAE={mae:.4f}  viés={np.mean(o - t):+.4f}")
        linhas.append((rotulo, dd, o, t, mae))
        todos_o += list(o); todos_t += list(t)

    r, p = stats.pearsonr(todos_o, todos_t)
    rs, ps = stats.spearmanr(todos_o, todos_t)
    mae_g = float(np.mean(np.abs(np.array(todos_o) - np.array(todos_t))))
    print(f"\nGLOBAL (n={len(todos_o)} pares): MAE={mae_g:.4f}  "
          f"Pearson r={r:.3f} (p={p:.1e})  Spearman={rs:.3f} (p={ps:.1e})")

    print("\n=== Representante kernel: SVC do TabZilla (só C ajustado) ===")
    for m, rot in KERNEL:
        dif = np.array([O[m][d] - T["SVM"][d] for d in DS])
        w = stats.wilcoxon(dif).pvalue
        print(f"   {rot:18s} ganho médio {np.mean(dif):+.4f}  "
              f"(vence em {int((dif > 0).sum())}/5, Wilcoxon p={w:.3f})")
        for d, x in zip(DS, dif):
            print(f"      {d:4s} SVM={T['SVM'][d]:.4f}  {m}={O[m][d]:.4f}  ({x:+.4f})")

    gera_tabela(O, T, linhas, mae_g, r, p)
    return 0


def gera_tabela(O, T, linhas, mae_g, r, p) -> None:
    def n(x):
        return f"{x:.4f}".replace(".", "{,}") if not np.isnan(x) else "---"

    corpo = []
    for rotulo, dd, o, t, mae in linhas:
        cel_o = " & ".join(n(O[[k for k, v, rr in PARES if rr == rotulo][0]][d])
                           if d in dd else "---" for d in DS)
        cel_t = " & ".join(n(t[dd.index(d)]) if d in dd else "---" for d in DS)
        corpo += [rf"    {rotulo} & este estudo & {cel_o} & --- \\",
                  rf"     & TabZilla & {cel_t} & {n(mae)} \\", r"    \addlinespace"]
    if corpo and corpo[-1] == r"    \addlinespace":
        corpo.pop()

    tex = "\n".join([
        r"\begin{table}[H]", r"  \centering",
        r"  \caption[Validação externa: concordância em AUC-ROC com o TabZilla]{Validação externa: AUC-ROC de teste neste estudo e no TabZilla "
        r"\cite{mcelfresh2023when} sobre os cinco conjuntos do Tier~1 presentes nos dois "
        r"levantamentos. Nos modelos homólogos, as duas implementações concordam "
        rf"(MAE global de ${n(mae_g)}$; $r$ de Pearson $= {n(r)}$, "
        rf"$p = {p * 1e7:.0f} \times 10^{{-7}}$), o que corrobora o \textit{{pipeline}} "
        r"por via externa. Só figuram as células em que os dois estudos avaliaram o "
        r"mesmo par modelo--conjunto; o TabZilla não reporta SAINT no GCR.}",
        r"  \label{tab:tabzilla}", r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{llrrrrrr}", r"    \toprule",
        r"    Modelo & Origem & " + " & ".join(DS) + r" & MAE \\",
        r"    \midrule", *corpo, r"    \bottomrule",
        r"  \end{tabular}}", r"\end{table}", ""])
    out = THESIS / "tables" / "tabzilla_agreement.tex"
    out.write_text(tex, encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    sys.exit(main())
