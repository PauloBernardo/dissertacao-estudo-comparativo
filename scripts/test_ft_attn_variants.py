"""
Teste rápido das 3 variantes de atenção inter-instâncias do FT-CUR:
  cur_full   — original: materializa A n×n (O(n²d)) → baseline de comparação
  nystrom    — opção 1 : Nyströmformer, C/R/W com softmax separado (O(nmd))
  linear_cur — opção 3 : kernel ELU+1 + CUR, sem softmax (O(nmd))

Roda 5 seeds em BCW, PID, GCR, TWS, TWC com hiperparâmetros fixos.
Objetivo: ver se as variantes são viáveis ou degradam completamente.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from src.models.ft_transformer_model import FTTransformerClassifier, fit_model
from src.models.landmark_selection import ColumnNormSelector
from src.data.loaders import DatasetLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASETS  = ["BCW", "PID", "GCR", "TWS", "TWC"]
SEEDS     = list(range(5))
MODES     = ["cur_full", "nystrom", "linear_cur"]
MODE_LABELS = {
    "cur_full":   "CUR-Full (original)",
    "nystrom":    "Nyström-Attn (opt.1)",
    "linear_cur": "Linear-CUR  (opt.3)",
}

# Hiperparâmetros fixos (médios do tuning anterior)
HP = dict(d_model=32, n_heads=2, n_layers=2, m_ratio=0.20,
          lr=3e-3, epochs=200, patience=20, weight_decay=1e-4,
          tau_ratio=0.10, val_fraction=0.20)


def run_one(X, y, seed: int, mode: str) -> float:
    rng = np.random.RandomState(seed)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed)
    sc = StandardScaler().fit(X_tr)
    X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)

    n = len(X_tr)
    n_val = max(1, round(HP["val_fraction"] * n))
    idx   = rng.permutation(n)
    X_val, X_fit = X_tr[idx[:n_val]], X_tr[idx[n_val:]]
    y_val, y_fit = y_tr[idx[:n_val]], y_tr[idx[n_val:]]

    m = max(2, round(HP["m_ratio"] * len(X_fit)))
    sel = ColumnNormSelector(n_landmarks=m, random_state=seed)
    sel.fit(X_fit)
    lm_idx = torch.tensor(sel.indices_, dtype=torch.long, device=DEVICE)

    model = FTTransformerClassifier(
        n_features=X.shape[1],
        d_model=HP["d_model"], n_heads=HP["n_heads"], n_layers=HP["n_layers"],
        use_inter_instance=True, tau_ratio=HP["tau_ratio"],
        attn_mode=mode,
    ).to(DEVICE)

    def t(arr, dtype=torch.float32):
        return torch.tensor(arr, dtype=dtype, device=DEVICE)

    fit_model(model, t(X_fit), t(y_fit.astype(np.float32)),
              t(X_val), t(y_val.astype(np.float32)),
              landmark_idx=lm_idx,
              lr=HP["lr"], epochs=HP["epochs"], patience=HP["patience"],
              weight_decay=HP["weight_decay"])

    model.eval()
    # inference: concatena X_tr completo como contexto
    n_ctx = len(X_tr)
    m_full = max(2, round(HP["m_ratio"] * n_ctx))
    sel2 = ColumnNormSelector(n_landmarks=m_full, random_state=seed)
    sel2.fit(X_tr)
    lm_full = torch.tensor(sel2.indices_, dtype=torch.long, device=DEVICE)

    X_ctx = np.concatenate([X_tr, X_te], axis=0)
    with torch.no_grad():
        logits = model(t(X_ctx), landmark_idx=lm_full)
    preds = (torch.sigmoid(logits[n_ctx:]).cpu().numpy() >= 0.5).astype(int)
    return f1_score(y_te, preds, average="macro", zero_division=0)


def main():
    results = defaultdict(lambda: defaultdict(list))

    for ds_name in DATASETS:
        print(f"\n{'='*55}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*55}")
        try:
            X, y, _ = DatasetLoader.load(ds_name)
            # Garantir labels 0/1
            uniq = np.unique(y)
            if set(uniq) == {-1, 1}:
                y = ((y + 1) // 2).astype(int)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        for mode in MODES:
            f1s = []
            for seed in SEEDS:
                try:
                    f1 = run_one(X, y, seed, mode)
                    f1s.append(f1)
                    print(f"  {MODE_LABELS[mode]:28s} seed={seed}  f1={f1:.4f}")
                except Exception as e:
                    print(f"  {MODE_LABELS[mode]:28s} seed={seed}  ERROR: {e}")
            if f1s:
                results[ds_name][mode] = f1s

    # ── Resumo ────────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"RESUMO — F1-macro médio (5 seeds)")
    print(f"{'='*72}")
    header = f"{'Dataset':<8}" + "".join(f"{MODE_LABELS[m]:>24}" for m in MODES)
    print(header)
    print("-" * 72)
    all_means = defaultdict(list)
    for ds in DATASETS:
        row = f"{ds:<8}"
        for mode in MODES:
            vals = results[ds].get(mode, [])
            if vals:
                mu = np.mean(vals)
                all_means[mode].append(mu)
                row += f"{mu:>24.4f}"
            else:
                row += f"{'---':>24}"
        print(row)
    print("-" * 72)
    row = f"{'Média':<8}"
    for mode in MODES:
        mu = np.mean(all_means[mode]) if all_means[mode] else float("nan")
        row += f"{mu:>24.4f}"
    print(row)

    # Delta vs cur_full
    print(f"\nDelta vs CUR-Full (original):")
    row = f"{'':8}"
    base = np.mean(all_means["cur_full"]) if all_means["cur_full"] else float("nan")
    for mode in MODES:
        mu = np.mean(all_means[mode]) if all_means[mode] else float("nan")
        delta = mu - base
        row += f"{delta:>+24.4f}"
    print(row)


if __name__ == "__main__":
    main()
