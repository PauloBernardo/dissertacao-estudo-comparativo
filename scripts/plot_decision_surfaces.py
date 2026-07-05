#!/usr/bin/env python3
"""Superfícies de decisão nos datasets artificiais 2D (TWS/TWM/TWC).

Gera duas figuras:
    fig6_decision_surfaces_lssvm.pdf/png
        3 LSSVMs (StandardLSSVM denso, Nystrom-SVM, DualFISTA) x 3 datasets,
        com os vetores-suporte/landmarks (subconjunto do treino com influência
        não-nula na predição) destacados por um anel preto.
    fig7_decision_surfaces_transformers.pdf/png
        2 Transformers (FT-Softmax inter-atributos, SAINT inter-instâncias) x
        3 datasets, só a superfície --- a esparsidade dos Transformers é de
        atenção, não amostral, então não há "vetor-suporte" análogo a destacar.

Hiperparâmetros: moda do GridSearchCV do Tier 1 (results/tier1_gridcv.json),
mesmo protocolo de split/escala usado nos experimentos (StandardScaler fit no
treino, split 70/30 estratificado, seed fixa).

Uso:
    python scripts/plot_decision_surfaces.py
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loaders import DatasetLoader
from src.data.preprocessing import _convert_labels, make_splits
from src.experiments.reproducibility import set_global_seed
from src.experiments.runner import _build_model
from src.tuning.grids import GRIDS

OUT = ROOT / "results" / "report_figs"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 0
DATASETS = ["TWS", "TWM", "TWC"]
DATASET_LABEL = {
    "TWS": "TWS (espirais)",
    "TWM": "TWM (luas)",
    "TWC": "TWC (tabuleiro)",
}

# Paleta validada (skill dataviz): categórico slot 1/6 (blue/red) — CVD-safe;
# divergente blue<->red com cinza neutro no meio (polaridade: classe -1 / 0 / +1).
BLUE = "#2a78d6"
RED = "#e34948"
GRAY = "#f0efec"
INK = "#0b0b0b"
CMAP = LinearSegmentedColormap.from_list("blue_gray_red", [BLUE, GRAY, RED], N=256)

LSSVM_MODELS = ["StandardLSSVM", "NystromLSSVMColnorm", "DualFISTA"]
LSSVM_LABEL = {
    "StandardLSSVM": "LSSVM-Std (denso)",
    "NystromLSSVMColnorm": "Nyström-SVM",
    "DualFISTA": "DualFISTA",
}
LSSVM_SIGNED = {"StandardLSSVM", "NystromLSSVMColnorm", "DualFISTA"}

TRANSFORMER_MODELS = ["FTTransformer_softmax", "SAINTColnorm"]
TRANSFORMER_LABEL = {
    "FTTransformer_softmax": "FT-Softmax (inter-atributos)",
    "SAINTColnorm": "SAINT (inter-instâncias)",
}

_TIER1_RESULTS = json.loads((ROOT / "results" / "tier1_gridcv.json").read_text())


def _modal_params(dataset: str, variant: str) -> dict:
    vals = [
        tuple(sorted(r["best_params"].items()))
        for r in _TIER1_RESULTS
        if r.get("status") == "ok" and r["variant"] == variant and r["dataset"] == dataset
    ]
    mode, _ = collections.Counter(vals).most_common(1)[0]
    return dict(mode)


def _label_format(variant: str) -> str:
    return "signed" if variant in LSSVM_SIGNED else "binary"


def _fit_model(variant: str, dataset: str, X_tr: np.ndarray, y_tr_raw: np.ndarray):
    set_global_seed(SEED)
    cfg = GRIDS[variant]
    fmt = _label_format(variant)
    params = {**cfg["fixed"], **_modal_params(dataset, variant)}
    model, _ = _build_model(cfg["model_name"], params, fmt)
    y_tr = _convert_labels(y_tr_raw, fmt)
    model.fit(X_tr, y_tr)
    return model


def _decision_surface(model, x_range, y_range, scaler, n=250):
    xx, yy = np.meshgrid(np.linspace(*x_range, n), np.linspace(*y_range, n))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_s = scaler.transform(grid)
    if hasattr(model, "decision_function"):
        Z = model.decision_function(grid_s)
    else:
        Z = model.predict_proba(grid_s)[:, 1] * 2 - 1
    return xx, yy, Z.reshape(xx.shape)


def _support_mask(model, variant: str, n_train: int) -> np.ndarray:
    """Máscara (sobre X_tr) dos pontos com influência não-nula na predição."""
    if variant == "NystromLSSVMColnorm":
        mask = np.zeros(n_train, dtype=bool)
        mask[model.support_vectors_] = True  # índices dos landmarks
        return mask
    return np.abs(model.alpha_) > model._ALPHA_ZERO_TOL  # BaseLSSVM subclasses


def _panel(ax, X_tr, y_tr_raw, xx, yy, Z, sv_mask, title, sparsity_pct=None):
    vmax = np.percentile(np.abs(Z), 97) + 1e-9
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    cf = ax.contourf(xx, yy, Z, levels=21, cmap=CMAP, norm=norm, alpha=0.85)
    # set_rasterized aqui warna "will be ignored" no QuadContourSet, mas reduz o
    # PDF de ~2.9MB para ~0.4MB (aplica às sub-coleções internas) — manter.
    cf.set_rasterized(True)
    ax.contour(xx, yy, Z, levels=[0.0], colors=INK, linewidths=1.3)

    neg_class = y_tr_raw.min()
    colors = np.where(y_tr_raw == neg_class, BLUE, RED)
    ax.scatter(X_tr[:, 0], X_tr[:, 1], c=colors, s=14, edgecolors="white",
               linewidths=0.4, zorder=3)

    if sv_mask is not None:
        ax.scatter(X_tr[sv_mask, 0], X_tr[sv_mask, 1], s=75, facecolors="none",
                   edgecolors=INK, linewidths=1.1, zorder=4)

    full_title = title if sparsity_pct is None else f"{title}\n({sparsity_pct:.0f}% esparso)"
    ax.set_title(full_title, fontsize=9.5)
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_grid(models: list[str], labels: dict, highlight_support: bool,
              out_name: str, fig_title: str):
    n_cols = len(models)
    fig, axes = plt.subplots(len(DATASETS), n_cols, figsize=(3.1 * n_cols, 3.3 * len(DATASETS)))
    if len(DATASETS) == 1:
        axes = axes[np.newaxis, :]

    for row, dataset in enumerate(DATASETS):
        X, y_raw, _ = DatasetLoader.load(dataset)
        X_tr, X_te, y_tr_raw, y_te_raw = make_splits(X, y_raw, test_size=0.30, seed=SEED)

        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)

        pad_x = 0.08 * (X[:, 0].max() - X[:, 0].min())
        pad_y = 0.08 * (X[:, 1].max() - X[:, 1].min())
        x_range = (X[:, 0].min() - pad_x, X[:, 0].max() + pad_x)
        y_range = (X[:, 1].min() - pad_y, X[:, 1].max() + pad_y)

        for col, variant in enumerate(models):
            model = _fit_model(variant, dataset, X_tr_s, y_tr_raw)
            xx, yy, Z = _decision_surface(model, x_range, y_range, scaler)

            sv_mask = None
            sparsity_pct = None
            if highlight_support:
                sv_mask = _support_mask(model, variant, len(X_tr))
                sparsity_pct = 100.0 * (1.0 - sv_mask.sum() / len(X_tr))

            title = labels[variant] if row == 0 else ""
            ax = axes[row, col]
            _panel(ax, X_tr, y_tr_raw, xx, yy, Z, sv_mask, title, sparsity_pct)
            if col == 0:
                ax.set_ylabel(DATASET_LABEL[dataset], fontsize=9.5)

    fig.suptitle(fig_title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / f"{out_name}.pdf", dpi=150)
    fig.savefig(OUT / f"{out_name}.png", dpi=150)
    plt.close(fig)
    print(f"Salvo: report_figs/{out_name}.pdf/png")


def main() -> None:
    _plot_grid(
        LSSVM_MODELS, LSSVM_LABEL, highlight_support=True,
        out_name="fig6_decision_surfaces_lssvm",
        fig_title="Superfície de Decisão e Vetores-Suporte/Landmarks --- LSSVMs",
    )
    _plot_grid(
        TRANSFORMER_MODELS, TRANSFORMER_LABEL, highlight_support=False,
        out_name="fig7_decision_surfaces_transformers",
        fig_title="Superfície de Decisão --- Transformers (esparsidade de atenção, não amostral)",
    )


if __name__ == "__main__":
    main()
