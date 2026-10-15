"""SAINT no embarque (2026-10-15): pesos + conjunto de treino como contexto (é o que a predição usa).
Mesma CPU, 1 thread, uma amostra por chamada; contexto N = 5000 (Ablação D) e N = 20000 (como os demais).
Treino curto (2 épocas): custo de predição depende só da arquitetura e do tamanho do contexto."""
import io, json, os, resource, sys, time
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, "../.."); sys.path.insert(0, "../../scripts"); sys.path.insert(0, ".")
import run_table18_benchmark as B
import run_tier2_gridcv as R2
R2.BUDGET_OVERRIDE.update(epochs=2, patience=2)
CFG = json.load(open("../../config/tier2_fixed_params.json"))["SAINTColnorm"]
ds, n = sys.argv[1], int(sys.argv[2])
os.environ["DATASET"] = ds
Xtr, Xte, ytr, yte = B.make_data(n, 0)
pipe, _ = R2._build_pipeline("SAINTColnorm", 0)
pipe.set_params(**{f"clf__{k}": v for k, v in CFG[ds.upper()].items()})
pipe.fit(Xtr, (ytr + 1) // 2)
clf = pipe.named_steps["clf"]
mods = [v for v in vars(clf).values() if isinstance(v, torch.nn.Module)]
buf = io.BytesIO(); torch.save([m.state_dict() for m in mods], buf)
w_kb = len(buf.getvalue()) / 1024
ctx_kb = clf.X_train_.astype(np.float32).nbytes / 1024
for i in range(3): pipe.predict(Xte[i:i + 1])
lat = []
for i in range(30):
    t = time.perf_counter(); pipe.predict(Xte[i:i + 1]); lat.append(time.perf_counter() - t)
print(json.dumps(dict(base=ds, n_ctx=n, weights_kb=round(w_kb, 1), context_kb=round(ctx_kb, 1),
                      lat_ms_median=round(float(np.median(lat)) * 1e3, 1), lat_ms_p99=round(float(np.percentile(lat, 99)) * 1e3, 1),
                      peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 0))))

# ── Embarque justo: só o último bloco importa para o ponto novo (lote B do treino/inferência) ──
B_ = clf.batch_size or 1024
ctx = clf._to_tensor(pipe.named_steps["scaler"].transform(Xtr)[-(B_ - 1):])   # B−1 linhas de treino
Xs = pipe.named_steps["scaler"].transform(Xte)
ref = pipe.predict(Xte[:200])
same, lat1 = [], []
with torch.no_grad():
    clf._model.eval()
    for i in range(200):
        x = clf._to_tensor(Xs[i:i + 1])
        t = time.perf_counter(); lg = clf._model(torch.cat([ctx, x]), landmark_idx=None)[-1]; lat1.append(time.perf_counter() - t)
        same.append(int(lg.item() >= 0) == ref[i])
print(json.dumps(dict(base=ds, n_ctx_saved=int(ctx.shape[0]), context_kb=round(ctx.numel() * 4 / 1024, 1),
                      weights_kb=round(w_kb, 1), lat_ms_median=round(float(np.median(lat1)) * 1e3, 1),
                      agree_with_published_path=float(np.mean(same)))))
