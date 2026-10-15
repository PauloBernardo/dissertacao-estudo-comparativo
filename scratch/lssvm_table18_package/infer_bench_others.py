"""Latência por amostra (1 por chamada) de XGBoost (.ubj) e Transformers (.pt TorchScript), processo novo,
mesma CPU e mesmo protocolo de infer_bench.py. torch e xgboost limitados a 1 thread (uma amostra não
paraleliza; evita que o tempo de criar threads domine). RAM: RSS depois de carregar o modelo menos RSS
depois de importar a biblioteca (o runtime em si é reportado à parte)."""
import json, os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np, psutil
sys.path.insert(0, "../.."); sys.path.insert(0, ".")
path, ds = sys.argv[1], sys.argv[2]
os.environ["DATASET"] = ds
import run_table18_benchmark as B
_, Xte, _, yte = B.make_data(20000, 0)
P = psutil.Process()
rss0 = P.memory_info().rss
if path.endswith(".ubj"):
    import xgboost as xgb
    rss1 = P.memory_info().rss
    t = time.perf_counter(); bst = xgb.Booster(); bst.load_model(path); bst.set_param({"nthread": 1})
    load_ms = (time.perf_counter() - t) * 1e3
    import pickle
    sc_ = None
    pred = lambda x: int(bst.inplace_predict(x)[0] >= 0.5)
    # o .ubj não guarda o StandardScaler; árvores são invariantes a escala monotônica por atributo,
    # mas o modelo foi treinado no dado escalado: refaz o scaler do treino (custo de 2·d operações, desprezível)
    from sklearn.preprocessing import StandardScaler
    Xtr, _, _, _ = B.make_data(20000, 0); sc = StandardScaler().fit(Xtr)
    Xin = sc.transform(Xte)
else:
    import torch
    torch.set_num_threads(1)
    rss1 = P.memory_info().rss
    t = time.perf_counter(); m = torch.jit.load(path).eval(); load_ms = (time.perf_counter() - t) * 1e3
    Xin = torch.tensor(Xte, dtype=torch.float32)
    def pred(x):
        with torch.no_grad():
            return int(m(x).item() >= 0)
rss2 = P.memory_info().rss
for i in range(50): pred(Xin[i:i + 1])
lat, yp = [], []
for i in range(len(Xin)):
    x = Xin[i:i + 1]
    t = time.perf_counter(); yp.append(pred(x)); lat.append(time.perf_counter() - t)
lat = np.array(lat) * 1e6
from sklearn.metrics import f1_score
print(json.dumps(dict(file=os.path.basename(path), file_kb=round(os.path.getsize(path) / 1024, 1),
                      load_ms=round(load_ms, 2), runtime_mb=round((rss1 - rss0) / 2**20, 1),
                      model_rss_kb=round((rss2 - rss1) / 1024, 1),
                      lat_us_median=round(float(np.median(lat)), 1), lat_us_p99=round(float(np.percentile(lat, 99)), 1),
                      f1_short_train=round(f1_score((yte + 1) // 2, yp, average="macro"), 4))))
