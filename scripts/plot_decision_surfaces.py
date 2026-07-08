#!/usr/bin/env python3
"""Superfícies de decisão nos datasets artificiais 2D (TWS/TWM/TWC), em N=400
(Tier 1) e N=2000 (Ablação A --- TWS_2k/TWM_2k/TWC_2k), para visualizar o ganho
dos Transformers ao escalar N.

Gera quatro figuras:
    fig6_decision_surfaces_lssvm_n400.pdf/png
    fig7_decision_surfaces_transformers_n400.pdf/png
    fig8_decision_surfaces_lssvm_n2000.pdf/png
    fig9_decision_surfaces_transformers_n2000.pdf/png

LSSVMs (com vetores-suporte/landmarks destacados por um anel preto):
    StandardLSSVM (denso), Nystrom-SVM, ADMM-Nystrom, DualFISTA.
Transformers (só superfície --- esparsidade de atenção, não amostral):
    FT-Softmax (inter-atributos), SAINT (inter-instâncias), FT-CUR (Nyströmformer).

Hiperparâmetros: moda do GridSearchCV oficial de cada regime
    N=400  -> results/tier1_gridcv.json
    N=2000 -> results/ablation_a_scaling.json (LSSVMs) +
              results/ablation_a_transformers.json (Transformers)
O ADMM-Nystrom não foi tunado oficialmente em N=2000 (ausente da Ablação A
original); um GridSearchCV leve e honesto é rodado aqui mesmo para esse caso,
já que sabemos (validação da Seção de Ablação D) que (lambda, tau) não
transferem entre escalas.

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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
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

# Paleta validada (skill dataviz): categórico slot 1/6 (blue/red) — CVD-safe;
# divergente blue<->red com cinza neutro no meio (polaridade: classe -1 / 0 / +1).
BLUE = "#2a78d6"
RED = "#e34948"
GRAY = "#f0efec"
INK = "#0b0b0b"
CMAP = LinearSegmentedColormap.from_list("blue_gray_red", [BLUE, GRAY, RED], N=256)

LSSVM_MODELS = ["StandardLSSVM", "NystromLSSVMColnorm", "ADMMNystromLSSVM", "DualFISTA"]
LSSVM_LABEL = {
    "StandardLSSVM": "LSSVM-Std (denso)",
    "NystromLSSVMColnorm": "Nyström-SVM",
    "ADMMNystromLSSVM": "ADMM-Nyström",
    "DualFISTA": "DualFISTA",
}
LSSVM_SIGNED = {"StandardLSSVM", "NystromLSSVMColnorm", "ADMMNystromLSSVM", "DualFISTA"}

TRANSFORMER_MODELS = ["FTTransformer_softmax", "SAINTColnorm", "FTTransformerCURColnorm"]
TRANSFORMER_LABEL = {
    "FTTransformer_softmax": "FT-Softmax (inter-atributos)",
    "SAINTColnorm": "SAINT (inter-instâncias)",
    "FTTransformerCURColnorm": "FT-CUR (Nyströmformer)",
}

DATASET_SETS = {
    "n400": {
        "datasets": ["TWS", "TWM", "TWC"],
        "labels": {
            "TWS": "TWS (espirais)\nN=400",
            "TWM": "TWM (luas)\nN=400",
            "TWC": "TWC (tabuleiro)\nN=400",
        },
        "results_files": ["tier1_gridcv.json"],
    },
    "n2000": {
        "datasets": ["TWS_2k", "TWM_2k", "TWC_2k"],
        "labels": {
            "TWS_2k": "TWS (espirais)\nN=2000",
            "TWM_2k": "TWM (luas)\nN=2000",
            "TWC_2k": "TWC (tabuleiro)\nN=2000",
        },
        "results_files": ["ablation_a_scaling.json", "ablation_a_transformers.json"],
    },
}


def _label_format(variant: str) -> str:
    return "signed" if variant in LSSVM_SIGNED else "binary"


def _load_results(files: list[str]) -> list[dict]:
    recs = []
    for f in files:
        p = ROOT / "results" / f
        if p.exists():
            recs.extend(r for r in json.loads(p.read_text()) if r.get("status") == "ok")
    return recs


def _modal_params(records: list[dict], dataset: str, variant: str) -> dict | None:
    vals = [
        tuple(sorted(r["best_params"].items()))
        for r in records
        if r["variant"] == variant and r["dataset"] == dataset
    ]
    if not vals:
        return None
    mode, _ = collections.Counter(vals).most_common(1)[0]
    return dict(mode)


def _quick_tune_admm_nystrom(dataset: str) -> dict:
    """GridSearchCV honesto para ADMM-Nystrom quando não há moda oficial no
    regime de N pedido (ex.: N=2000 não fez parte da Ablação A original).
    Necessário porque sabemos (Ablação D) que (lambda, tau) não transferem
    entre escalas --- reusar a moda de outro N seria repetir o mesmo erro.
    """
    X, y_raw, _ = DatasetLoader.load(dataset)
    X_tr, _, y_tr_raw, _ = make_splits(X, y_raw, test_size=0.30, seed=SEED)
    y_tr = _convert_labels(y_tr_raw, "signed")

    cfg = GRIDS["ADMMNystromLSSVM"]
    estimator, _ = _build_model(cfg["model_name"], dict(cfg["fixed"]), "signed")
    estimator.set_params(random_state=SEED)
    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
    param_grid = {f"clf__{k}": v for k, v in cfg["grid"].items()}

    search = GridSearchCV(
        pipeline, param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        scoring="f1_macro", n_jobs=-1, error_score=0.0,
    )
    search.fit(X_tr, y_tr)
    print(f"  [quick-tune] ADMM-Nystrom @ {dataset}: {search.best_params_} (f1_cv={search.best_score_:.3f})")
    return {k.replace("clf__", ""): v for k, v in search.best_params_.items()}


def _get_params(dataset: str, variant: str, set_key: str, records: list[dict]) -> dict:
    cfg = GRIDS[variant]
    modal = _modal_params(records, dataset, variant)
    if modal is None:
        if variant == "ADMMNystromLSSVM":
            modal = _quick_tune_admm_nystrom(dataset)
        else:
            raise RuntimeError(f"Sem hiperparâmetros para {variant} em {dataset} ({set_key})")
    return {**cfg["fixed"], **modal}


def _fit_model(variant: str, dataset: str, set_key: str, records: list[dict],
              X_tr: np.ndarray, y_tr_raw: np.ndarray):
    set_global_seed(SEED)
    cfg = GRIDS[variant]
    fmt = _label_format(variant)
    params = _get_params(dataset, variant, set_key, records)
    model, _ = _build_model(cfg["model_name"], params, fmt)
    if hasattr(model, "get_params") and "random_state" in model.get_params():
        model.set_params(random_state=SEED)
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


def _support_points(model, variant: str, X_tr_scaled: np.ndarray) -> np.ndarray:
    """Retorna as COORDENADAS (espaço escalado) dos pontos com influência
    não-nula na predição --- vetor-suporte (primal/dual) ou landmark (Nyström).
    """
    if variant == "NystromLSSVMColnorm":
        return X_tr_scaled[model.support_vectors_]  # índices dos landmarks
    if variant == "ADMMNystromLSSVM":
        nz = np.abs(model.theta_) > 1e-6
        return model.landmarks_[nz]  # já são coordenadas (cópia de X no fit)
    # BaseLSSVM subclasses (StandardLSSVM, DualFISTA): alpha_ não-nulo
    nz = np.abs(model.alpha_) > model._ALPHA_ZERO_TOL
    return X_tr_scaled[nz]


def _panel(ax, X_tr, y_tr_raw, xx, yy, Z, sv_points_scaled, scaler, title, sparsity_pct=None):
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

    if sv_points_scaled is not None:
        sv_points = scaler.inverse_transform(sv_points_scaled)
        ax.scatter(sv_points[:, 0], sv_points[:, 1], s=75, facecolors="none",
                   edgecolors=INK, linewidths=1.1, zorder=4)

    full_title = title if sparsity_pct is None else f"{title}\n({sparsity_pct:.0f}% esparso)"
    ax.set_title(full_title, fontsize=9.5)
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_grid(set_key: str, models: list[str], labels: dict, highlight_support: bool,
              out_name: str, fig_title: str):
    dset = DATASET_SETS[set_key]
    datasets = dset["datasets"]
    records = _load_results(dset["results_files"])

    n_cols = len(models)
    fig, axes = plt.subplots(len(datasets), n_cols, figsize=(3.1 * n_cols, 3.5 * len(datasets)))
    if len(datasets) == 1:
        axes = axes[np.newaxis, :]

    for row, dataset in enumerate(datasets):
        X, y_raw, _ = DatasetLoader.load(dataset)
        X_tr, X_te, y_tr_raw, y_te_raw = make_splits(X, y_raw, test_size=0.30, seed=SEED)

        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)

        pad_x = 0.08 * (X[:, 0].max() - X[:, 0].min())
        pad_y = 0.08 * (X[:, 1].max() - X[:, 1].min())
        x_range = (X[:, 0].min() - pad_x, X[:, 0].max() + pad_x)
        y_range = (X[:, 1].min() - pad_y, X[:, 1].max() + pad_y)

        for col, variant in enumerate(models):
            print(f"[{out_name}] {dataset} / {variant} ...")
            model = _fit_model(variant, dataset, set_key, records, X_tr_s, y_tr_raw)
            xx, yy, Z = _decision_surface(model, x_range, y_range, scaler)

            sv_points_scaled = None
            sparsity_pct = None
            if highlight_support:
                sv_points_scaled = _support_points(model, variant, X_tr_s)
                sparsity_pct = 100.0 * (1.0 - len(sv_points_scaled) / len(X_tr))

            title = labels[variant] if row == 0 else ""
            ax = axes[row, col]
            _panel(ax, X_tr, y_tr_raw, xx, yy, Z, sv_points_scaled, scaler, title, sparsity_pct)
            if col == 0:
                ax.set_ylabel(dset["labels"][dataset], fontsize=9.5)

    fig.suptitle(fig_title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / f"{out_name}.pdf", dpi=150)
    fig.savefig(OUT / f"{out_name}.png", dpi=150)
    plt.close(fig)
    print(f"Salvo: report_figs/{out_name}.pdf/png")


def main() -> None:
    for set_key, n_label, lssvm_name, tf_name in [
        ("n400", "N=400", "fig6_decision_surfaces_lssvm_n400", "fig7_decision_surfaces_transformers_n400"),
        ("n2000", "N=2000", "fig8_decision_surfaces_lssvm_n2000", "fig9_decision_surfaces_transformers_n2000"),
    ]:
        _plot_grid(
            set_key, LSSVM_MODELS, LSSVM_LABEL, highlight_support=True,
            out_name=lssvm_name,
            fig_title=f"Superfície de Decisão e Vetores-Suporte/Landmarks --- LSSVMs ({n_label})",
        )
        _plot_grid(
            set_key, TRANSFORMER_MODELS, TRANSFORMER_LABEL, highlight_support=False,
            out_name=tf_name,
            fig_title=f"Superfície de Decisão --- Transformers ({n_label}, esparsidade de atenção)",
        )


if __name__ == "__main__":
    main()
