"""Inferência 'na prática' a partir do .npz, num processo novo por modelo (2026-10-15).

Uso: infer_bench.py <arquivo.npz> <base>
Mede: tempo de carga, memória alocada pelo modelo carregado (tracemalloc), latência por amostra
(1 amostra por chamada, como num ciclo de CLP; mediana e p99 de 2000 chamadas), pico de memória
de UMA predição e concordância com a predição em lote.
"""
import json, os, sys, time, tracemalloc
import numpy as np
sys.path.insert(0, "../.."); sys.path.insert(0, ".")

path, ds = sys.argv[1], sys.argv[2]
os.environ["DATASET"] = ds
import run_table18_benchmark as B
_, Xte, _, yte = B.make_data(20000, 0)


def predict_one(d, x):
    xs = (x - d["mu"]) / d["s"]
    k = np.exp(-((d["Z"] - xs) ** 2).sum(1) / (2 * d["sigma"] ** 2))
    if "L" in d:
        L, phi = d["L"], np.empty_like(k)
        for j in range(len(k)):
            phi[j] = (k[j] - phi[:j] @ L[j, :j]) / L[j, j]
        k = phi
    return 1 if k @ d["w"] + d["b"] >= 0 else -1


tracemalloc.start()
t = time.perf_counter(); d = {k: v for k, v in np.load(path).items()}; load_ms = (time.perf_counter() - t) * 1e3
model_kb = tracemalloc.get_traced_memory()[0] / 1024
for x in Xte[:50]: predict_one(d, x)                     # aquecimento
tracemalloc.reset_peak(); base = tracemalloc.get_traced_memory()[0]
predict_one(d, Xte[0]); peak_kb = (tracemalloc.get_traced_memory()[1] - base) / 1024
tracemalloc.stop()

lat, yp = [], []
for x in Xte:
    t = time.perf_counter(); yp.append(predict_one(d, x)); lat.append(time.perf_counter() - t)
lat = np.array(lat) * 1e6
from sklearn.metrics import f1_score
print(json.dumps(dict(file=os.path.basename(path), file_kb=round(os.path.getsize(path) / 1024, 1),
                      load_ms=round(load_ms, 2), model_ram_kb=round(model_kb, 1), pred_peak_kb=round(peak_kb, 1),
                      lat_us_median=round(float(np.median(lat)), 1), lat_us_p99=round(float(np.percentile(lat, 99)), 1),
                      f1=round(f1_score(yte, yp, average="macro"), 4))))
