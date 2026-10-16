"""Tamanho do modelo para predição (float64), a partir dos pontos guardados (2026-10-15).

Nyström/ADMM/FISTA/PCP: pts·d (landmarks ou pivôs) + pts (pesos); o PCP fiel (Zhou 2016) não guarda L.
Todos: +1 (bias) + 2d (StandardScaler).
"""
import json, sys
D = {"synthetic": 20, "adult": 14, "credit": 23, "bank": 16}


def kb(variant: str, pts: float, d: int) -> float:
    n = pts * d + pts + 1 + 2 * d     # 2026-10-15: PCP fiel prediz com α_B (sem o triângulo de L)
    return n * 8 / 1024


if __name__ == "__main__":
    for path, ds in [(p, p.split("/")[-1].split("_")[0]) for p in sys.argv[1:]]:
        d = D.get(ds, 20)
        rs = json.load(open(path))
        for r in rs:
            if not r.get("skipped") and r.get("n_support_median") is not None:
                r["model_kb"] = round(kb(r["variant"], r["n_support_median"], d), 1)
        json.dump(rs, open(path, "w"), indent=2)
        print(path, "d=", d, [(r["variant"][:5], r["n"], r.get("model_kb")) for r in rs if r["n"] == 20000])
