#!/usr/bin/env python3
"""Tabela de custo de embarque (2026-10-15): tamanho do arquivo exportado e latência por amostra.

Tudo medido na MESMA CPU (GPU desligada), uma amostra por chamada, processo novo por modelo,
N = 20 mil de treino e 2000 de teste em ADULT, CREDIT e BANK. Fontes (scratch/lssvm_table18_package/saved_models/):
  infer.jsonl          LSSVMs, export mínimo .npz, predição em numpy   (infer_bench.py)
  infer_others.jsonl   XGBoost (.ubj) e FT-Softmax / FT-CUR streaming (TorchScript)  (infer_bench_others.py)
  saint.jsonl          SAINT com um lote de contexto (B−1 linhas de treino)  (saint_embed.py)
Saída: dissertacao-latex/tables/embedding_cost.tex
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SM = ROOT / "scratch/lssvm_table18_package/saved_models"
OUT = ROOT.parent / "dissertacao-latex/tables/embedding_cost.tex"
BASES = ["adult", "credit", "bank"]
ROWS = [("admm_m1", r"ADMM-Nyström ($m/N=0{,}01$)"), ("admm_m30", r"ADMM-Nyström ($m/N=0{,}3$)"),
        ("fista_m1", r"FISTA-Nyström ($m/N=0{,}01$)"), ("fista_m30", r"FISTA-Nyström ($m/N=0{,}3$)"),
        ("nys_m1", r"Nyström-SVM ($m/N=0{,}01$)"), ("nys_m30", r"Nyström-SVM ($m/N=0{,}3$)"),
        ("pcp", "LSSVM-PCP"), ("xgb", "XGBoost"), ("ftsoftmax", "FT-Softmax"),
        ("ftcur", r"FT-CUR (\textit{streaming})"), ("saint", r"SAINT (um lote de contexto)")]


def load():
    d = {}
    for f in ("infer.jsonl", "infer_others.jsonl"):
        for line in (SM / f).read_text().splitlines():
            r = json.loads(line); base, model = r["file"].split(".")[0].split("_", 1)
            d[(model, base)] = (r["file_kb"], r["lat_us_median"])
    for line in (SM / "saint.jsonl").read_text().splitlines():
        r = json.loads(line)
        d[("saint", r["base"])] = (r["weights_kb"] + r["context_kb"], r["lat_ms_median"] * 1e3)
    return d


def fk(kb):
    s = f"{kb:,.0f}" if kb >= 100 else f"{kb:.1f}"
    return s.replace(",", r"\,").replace(".", "{,}")


def fl(us):
    return (f"{us/1000:.1f}".replace(".", "{,}") + r"\,ms") if us >= 1000 else f"{us:.0f}"


def main():
    d = load()
    L = [r"\begin{table}[H]", r"\centering",
         r"\caption[Custo de embarque: tamanho do modelo e latência por amostra]{Custo de embarque com $N=20\,000$: tamanho do arquivo exportado (KB) e latência mediana de predição por amostra ($\mu$s), uma amostra por chamada, todos na mesma CPU. LSSVMs em \textit{numpy} (só os pontos com peso não nulo, os pesos e o \textit{scaler}); XGBoost no formato nativo; Transformers em TorchScript. O F1 de referência é o publicado (Tier~2 e Ablação~D), não o desta medição.}",
         r"\label{tab:embedding_cost}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l" + "rr" * 3 + "}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{b.upper()}}}" for b in BASES) + r" \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         "Modelo" + r" & KB & $\mu$s" * 3 + r" \\", r"\midrule"]
    for i, (k, lab) in enumerate(ROWS):
        if k in ("xgb",): L.append(r"\midrule")
        L.append(lab + "".join(f" & {fk(d[(k, b)][0])} & {fl(d[(k, b)][1])}" for b in BASES) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    OUT.write_text("\n".join(L) + "\n"); print("->", OUT)
    for k, lab in ROWS: print(f"{k:10s}", [d[(k, b)] for b in BASES])


if __name__ == "__main__":
    main()
