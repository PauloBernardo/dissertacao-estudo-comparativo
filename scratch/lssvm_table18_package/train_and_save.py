"""Treina e salva os modelos com N = 20 mil (ADULT, CREDIT, BANK) para medir o embarque na prática (2026-10-15).

Para cada modelo grava:
  <base>_<modelo>.pkl  pipeline completo (scaler + modelo), como sai do scikit-learn
  <base>_<modelo>.npz  só o que a predição usa, em float64 (formato de embarque)
e confere que a predição a partir do .npz é idêntica à do pipeline.
"""
import json, os, pickle, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "../.."); sys.path.insert(0, ".")
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import run_table18_benchmark as B
from src.models.nystrom_lssvm_wrapper import NystromLSSVMRandom
from src.models.lssvm.primal.admm_nystrom import ADMMNystromLSSVM
from src.models.lssvm.primal.fista_nystrom import FISTANystromLSSVM
from src.models.lssvm.primal.pcp_lssvm import PCPLSSVm

OUT = Path("saved_models")
MODELS = {
    "nys_m1":    lambda: NystromLSSVMRandom(sigma=5.0, gamma=10.0, m_ratio=0.01, random_state=0),
    "admm_m1":   lambda: ADMMNystromLSSVM(sigma=5.0, tau=0.1, lambda_rel=0.001, kernel_reg=True, m_ratio=0.01, landmark_method="random", rho="adaptive", max_iter=500, random_state=0),
    "fista_m1":  lambda: FISTANystromLSSVM(sigma=5.0, tau=0.1, lambda_rel=0.001, kernel_reg=True, m_ratio=0.01, landmark_method="random", max_iter=5000, random_state=0),
    "nys_m30":   lambda: NystromLSSVMRandom(sigma=5.0, gamma=10.0, m_ratio=0.30, random_state=0),
    "admm_m30":  lambda: ADMMNystromLSSVM(sigma=5.0, tau=0.1, lambda_rel=0.001, kernel_reg=True, m_ratio=0.30, landmark_method="random", rho="adaptive", max_iter=500, random_state=0),
    "fista_m30": lambda: FISTANystromLSSVM(sigma=5.0, tau=0.1, lambda_rel=0.001, kernel_reg=True, m_ratio=0.30, landmark_method="random", max_iter=5000, random_state=0),
    "pcp":       lambda: PCPLSSVm(sigma=5.0, tau=0.1, rank=200),
}


def export(pipe) -> dict:
    """Arrays mínimos: f(x) = Σ_j w_j k(z_j, (x−μ)/s) + b, mais o bloco L no PCP."""
    sc, m = pipe.named_steps["scaler"], pipe.named_steps["clf"]
    d = dict(mu=sc.mean_, s=sc.scale_, sigma=np.float64(m.sigma if hasattr(m, "sigma") else m._model.kernel.sigma))
    if isinstance(m, NystromLSSVMRandom):
        d.update(Z=m._model.nystrom_.landmarks_, w=m._model._w_alpha_, b=np.float64(m._model.b_))
    elif isinstance(m, PCPLSSVm):   # 2026-10-15: PCP fiel (Zhou 2016) prediz com α_B, sem L
        d.update(Z=m.X_pivots_, w=m.coef_, b=np.float64(m.bias_))
    else:
        d.update(Z=m._landmarks_nz_, w=m._theta_nz_, b=np.float64(m.bias_))
    return d


def predict_npz(d: dict, X: np.ndarray) -> np.ndarray:
    """Predição só com numpy, a partir do .npz (o que rodaria no alvo embarcado)."""
    Xs = (X - d["mu"]) / d["s"]
    K = np.exp(-((Xs[:, None, :] - d["Z"][None]) ** 2).sum(-1) / (2 * d["sigma"] ** 2))
    if "L" in d:   # PCP: substituição direta no bloco triangular
        L, phi = d["L"], np.zeros_like(K)
        for j in range(L.shape[0]):
            phi[:, j] = (K[:, j] - phi[:, :j] @ L[j, :j]) / L[j, j]
        K = phi
    s = K @ d["w"] + d["b"]
    return np.where(s >= 0, 1, -1)


if __name__ == "__main__":
    rows = []
    for ds in ("adult", "credit", "bank"):
        os.environ["DATASET"] = ds
        Xtr, Xte, ytr, yte = B.make_data(20000, 0)
        for name, mk in MODELS.items():
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", mk())])
            t = time.time(); pipe.fit(Xtr, ytr); fit_s = time.time() - t
            yp = pipe.predict(Xte)
            pkl = OUT / f"{ds}_{name}.pkl"; pkl.write_bytes(pickle.dumps(pipe))
            d = export(pipe); npz = OUT / f"{ds}_{name}.npz"; np.savez(npz, **d)
            same = bool(np.array_equal(predict_npz(dict(np.load(npz)), Xte), yp))
            rows.append(dict(base=ds, model=name, fit_s=round(fit_s, 2), f1=round(f1_score(yte, yp, average="macro"), 4),
                             pts=int(len(d["w"])), pkl_kb=round(pkl.stat().st_size / 1024, 1),
                             npz_kb=round(npz.stat().st_size / 1024, 1), npz_pred_equal=same))
            print(rows[-1], flush=True)
            json.dump(rows, open(OUT / "summary.json", "w"), indent=2)
