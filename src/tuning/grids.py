"""Hyperparameter grids for the Tier 1 grid-search protocol.

Each entry maps a variant_name (the key used in the output JSON) to:
    model_name : runner_name passed to ``src.experiments.runner._build_model``.
                 Two variants may share the same model_name (e.g.,
                 ``ADMMNesterovLSSVM`` and ``ADMMElasticNet`` both build
                 the same class, differing only in ``lambda2_``).
    grid       : ``param_grid`` for ``sklearn.GridSearchCV``. Values are
                 lists; the Cartesian product defines the search space.
    fixed      : kwargs passed at estimator instantiation but NOT varied.
                 Architectural defaults (Gorishniy 2021 for Transformers)
                 and auto-set parameters (e.g., ``rho=None`` for ADMM)
                 live here.
    needs_gpu  : True for PyTorch-backed estimators; tells the runner to
                 use ``n_jobs=1`` so concurrent GridSearchCV folds don't
                 fight over CUDA memory.

References for the grids:
    - LSSVM σ, τ : Suykens et al. 2002 (LSSVM book) + Marinho et al.
                   observed optima for ADMM-N on Tier 1 datasets.
    - ADMM-N λ   : boundary analysis showed 73% choosing 0.01 (min);
                   shifted to [0.001, 0.005, 0.01, 0.1] — larger values
                   (1.0, 10.0) discarded (rarely chosen, lower F1).
    - FISTA λ    : same analysis — 68–86% at 0.01; shifted down similarly.
    - FT-Transformer fixed defaults : Gorishniy et al. 2021, Table 12.
"""

from __future__ import annotations


GRIDS: dict[str, dict] = {

    # ──────────────── ADMM-Nyström: esparso + escalável ──────────────────

    "ADMMNystromLSSVM": {
        # Modo A — single-machine.  Grade: sigma/tau/lambda_ livres, m_ratio fixo.
        "model_name": "ADMMNystromLSSVM",
        "grid": {
            "sigma":   [0.1, 0.5, 2.0],
            "tau":     [0.005, 0.05, 0.5, 5.0, 50.0],
            "lambda_": [1.0, 0.1, 0.01],
        },
        "fixed": {"m_ratio": 0.30, "rho": None, "max_iter": 500,
                  "landmark_method": "colnorm", "n_blocks": 1, "n_jobs": 1},
        "needs_gpu": False,
    },

    "ADMMNystromDistributed": {
        # Modo B — block-parallel (4 blocos, todos os cores disponíveis).
        # Mesma grade que A; n_blocks/n_jobs ativam o modo paralelo.
        "model_name": "ADMMNystromLSSVM",
        "grid": {
            "sigma":   [0.1, 0.5, 2.0],
            "tau":     [0.005, 0.05, 0.5, 5.0, 50.0],
            "lambda_": [1.0, 0.1, 0.01],
        },
        "fixed": {"m_ratio": 0.30, "rho": None, "max_iter": 500,
                  "landmark_method": "colnorm", "n_blocks": 4, "n_jobs": 4},
        "needs_gpu": False,
    },

    "FISTANystrom": {
        # FISTA primal no espaço Nyström — mesmo espaço que ADMMNystromLSSVM
        # mas sem parâmetro ρ. Mesma grade de sigma/tau/lambda_.
        "model_name": "FISTANystromLSSVM",
        "grid": {
            "sigma":   [0.1, 0.5, 2.0],
            "tau":     [0.005, 0.05, 0.5, 5.0, 50.0],
            "lambda_": [1.0, 0.1, 0.01],
        },
        "fixed": {"m_ratio": 0.30, "landmark_method": "colnorm", "max_iter": 5000},
        "needs_gpu": False,
    },

    # ───────────────────────── LSSVM baselines ────────────────────────────

    "StandardLSSVM": {
        "model_name": "StandardLSSVM",
        "grid": {
            "sigma": [0.05, 0.15, 0.5, 1.5, 5.0],
            "tau":   [0.1, 0.5, 2.5, 12.5, 50.0],
        },
        "fixed": {},
        "needs_gpu": False,
    },

    "PCPLSSVm": {
        # sigma=0.15 em 100% dos casos sintéticos (TWS/TWM/TWC Tier 1).
        "model_name": "PCPLSSVm",
        "grid": {
            "tau":  [0.1, 0.5, 2.5, 12.5, 50.0],
            "rank": [20, 50, 100, 200],
        },
        "fixed": {"sigma": 0.15},
        "needs_gpu": False,
    },

    "FSALSSVm": {
        # sigma=0.5 em 73% dos casos sintéticos (TWS/TWM/TWC Tier 1).
        "model_name": "FSALSSVm",
        "grid": {
            "tau":          [0.1, 0.5, 2.5, 12.5, 50.0],
            "n_components": [50, 100, 200],
        },
        "fixed": {"sigma": 0.5},
        "needs_gpu": False,
    },

    "IPLSSVm": {
        # sigma=0.5 em 83% dos casos sintéticos (TWS/TWM/TWC Tier 1).
        "model_name": "IPLSSVm",
        "grid": {
            "tau":             [0.1, 0.5, 2.5, 12.5, 50.0],
            "selection_ratio": [0.1, 0.2, 0.5, 0.7],
        },
        "fixed": {"sigma": 0.5},
        "needs_gpu": False,
    },

    "PruningLSSVM": {
        # pruning_rate=0.05 em 100% dos casos sintéticos (TWS/TWM/TWC Tier 1).
        "model_name": "PruningLSSVM",
        "grid": {
            "sigma": [0.05, 0.15, 0.5, 1.5, 5.0, 15.0],
            "tau":   [0.1, 0.5, 2.5, 12.5, 50.0],
        },
        "fixed": {"drop_tolerance": 0.05, "pruning_rate": 0.05},
        "needs_gpu": False,
    },

    "OppositeMapsLSSVM": {
        # sigma=0.5 em 87%, n_prototypes=100 em 97% dos casos sintéticos.
        "model_name": "OppositeMapsLSSVM",
        "grid": {
            "tau": [0.1, 0.5, 2.5, 12.5, 50.0],
        },
        "fixed": {"drop_tolerance": 0.05, "sigma": 0.5, "n_prototypes": 100},
        "needs_gpu": False,
    },

    # ───────────────────── LSSVM paper-base + variants ────────────────────

    "ADMMNesterovLSSVM": {
        # HAB probe (30 seeds): lambda_=1.0 venceu em 57%, sigma=0.5 em 97%.
        # sigma [0.1, 0.5, 2.0]: extremos 8.0/32.0 descartados (nunca escolhidos na probe).
        "model_name": "ADMMNesterovLSSVM",
        "grid": {
            "sigma":   [0.1, 0.5, 2.0],
            "tau":     [0.005, 0.05, 0.5, 5.0, 50.0],
            "lambda_": [1.0, 0.1, 0.01],
        },
        "fixed": {"rho": None, "max_iter": 500},
        "needs_gpu": False,
    },

    "ADMMElasticNet": {
        # Mesmo protocolo que ADMMNesterovLSSVM; lambda2_ fixo (L2 ridge não afeta esparsidade).
        "model_name": "ADMMNesterovLSSVM",
        "grid": {
            "sigma":   [0.1, 0.5, 2.0],
            "tau":     [0.005, 0.05, 0.5, 5.0, 50.0],
            "lambda_": [1.0, 0.1, 0.01],
        },
        "fixed": {"lambda2_": 0.001, "rho": None, "max_iter": 500},
        "needs_gpu": False,
    },

    "FISTANesterov": {
        # sigma livre (3 pontos); lambda_ estendido: 0.001 recupera BCW/GCR, 1.0 captura regime esparso.
        "model_name": "FISTANesterovLSSVM",
        "grid": {
            "sigma":   [0.1, 0.5, 2.0],
            "tau":     [0.01, 0.1, 1.0, 10.0],
            "lambda_": [1.0, 0.1, 0.01, 0.001],
        },
        "fixed": {},
        "needs_gpu": False,
    },

    "DualFISTA": {
        # sigma livre (3 pontos); lambda_ estendido para capturar regime esparso.
        "model_name": "DualFISTALSSVM",
        "grid": {
            "sigma":   [0.1, 0.5, 2.0],
            "tau":     [0.01, 0.1, 1.0, 10.0],
            "lambda_": [1.0, 0.1, 0.01],
        },
        "fixed": {},
        "needs_gpu": False,
    },

    "NystromLSSVMColnorm": {
        # sigma=0.5 em 100%, m_ratio=0.30 em 78% dos casos sintéticos.
        "model_name": "NystromLSSVMColnorm",
        "grid": {
            "gamma": [0.1, 1.0, 10.0, 30.0, 50.0, 100.0],
        },
        "fixed": {"sigma": 0.5, "m_ratio": 0.30},
        "needs_gpu": False,
    },

    # ─────────────────────────── XGBoost ──────────────────────────────────

    "XGBoost": {
        "model_name": "XGBoost",
        "grid": {
            "n_estimators":  [100, 300],
            "max_depth":     [3, 6, 9],
            "learning_rate": [0.01, 0.1, 0.3],
        },
        "fixed": {},
        "needs_gpu": False,
    },

    # ───────────────── FT-Transformer baselines (4 variants) ──────────────

    "FTTransformer_softmax": {
        "model_name": "FTTransformer",
        "grid": {
            "num_blocks": [2, 3, 4],
            "num_heads":  [2, 4],
        },
        "fixed": {
            "attention_type": "softmax",
            "embedding_dim":  64,
            "dropout":        0.1,
            "lr":             1e-3,
            "batch_size":     256,
            "max_epochs":     40,
            "patience":       6,
        },
        "needs_gpu": True,
    },

    "FTTransformer_topk": {
        "model_name": "FTTransformer",
        "grid": {
            "num_blocks": [2, 3, 4],
            "num_heads":  [2, 4],
        },
        "fixed": {
            "attention_type": "topk",
            "topk_ratio":     0.10,
            "embedding_dim":  64,
            "dropout":        0.1,
            "lr":             1e-3,
            "batch_size":     256,
            "max_epochs":     40,
            "patience":       6,
        },
        "needs_gpu": True,
    },

    "FTTransformer_entmax": {
        "model_name": "FTTransformer",
        "grid": {
            "num_blocks": [2, 3, 4],
            "num_heads":  [2, 4],
        },
        "fixed": {
            "attention_type": "entmax",
            "alpha":          1.5,
            "embedding_dim":  64,
            "dropout":        0.1,
            "lr":             1e-3,
            "batch_size":     256,
            "max_epochs":     40,
            "patience":       6,
        },
        "needs_gpu": True,
    },

    "FTTransformer_sparsemax": {
        "model_name": "FTTransformer",
        "grid": {
            "num_blocks": [2, 3, 4],
            "num_heads":  [2, 4],
        },
        "fixed": {
            "attention_type": "sparsemax",
            "embedding_dim":  64,
            "dropout":        0.1,
            "lr":             1e-3,
            "batch_size":     256,
            "max_epochs":     40,
            "patience":       6,
        },
        "needs_gpu": True,
    },

    # ──────────────── SAINT + FT-CUR (inter-instance attention) ───────────

    "SAINTColnorm": {
        "model_name": "SAINTColnorm",
        "grid": {
            "n_layers": [1, 2, 3],
            "n_heads":  [2, 4],
        },
        "fixed": {
            "d_model":            32,
            "lr":                 1e-3,
            "epochs":             40,
            "patience":           6,
            "early_stop_metric":  "val_loss",
        },
        "needs_gpu": True,
    },

    "FTTransformerCURColnorm": {
        "model_name": "FTTransformerCURColnorm",
        "grid": {
            "n_layers": [1, 2, 3],
            "n_heads":  [2, 4],
            "m_ratio":  [0.10, 0.20],
        },
        "fixed": {
            "d_model":            32,
            "lr":                 1e-3,
            "epochs":             40,
            "patience":           6,
            "early_stop_metric":  "val_loss",
        },
        "needs_gpu": True,
    },
}


def grid_size(variant: str) -> int:
    """Number of grid points (Cartesian product) for a variant."""
    g = GRIDS[variant]["grid"]
    n = 1
    for v in g.values():
        n *= len(v)
    return n
