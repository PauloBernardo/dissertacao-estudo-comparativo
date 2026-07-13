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
        # sigma precisa ser buscado: kernel RBF exato colapsa para zero em dados UCI
        # com muitas features (||x-y||²>>σ²). FSALSSVm/IPLSSVm usam RFF (não afetados).
        "model_name": "PCPLSSVm",
        "grid": {
            "sigma": [0.5, 1.5, 5.0],
            "tau":   [0.1, 0.5, 2.5, 12.5, 50.0],
            "rank":  [20, 50, 100, 200],
        },
        "fixed": {},
        "needs_gpu": False,
    },

    "FSALSSVm": {
        # ⚠️ DEPRECADO (2026-07-10): Matching Pursuit, não o backfitting de
        # Jiao. Substituído por FSALSSVmOriginal. Histórico.
        # sigma no grid: RFF de alta frequência (sigma pequeno) não captura estrutura
        # nos dados UCI com muitas features — mesmo problema do PCPLSSVm.
        "model_name": "FSALSSVm",
        "grid": {
            "sigma":        [0.5, 1.5, 5.0],
            "tau":          [0.1, 0.5, 2.5, 12.5, 50.0],
            "n_components": [50, 100, 200],
        },
        "fixed": {},
        "needs_gpu": False,
    },

    "FSALSSVmOriginal": {
        # Reprodução FIEL do FSALS-SVM (Jiao et al. 2007): critério de resíduo
        # quadrático normalizado (Eq. 30) + backfitting. Nº de basis functions
        # como fração de N (escala com o dataset).
        "model_name": "FSALSSVmOriginal",
        "grid": {
            "sigma":   [0.5, 1.5, 5.0],
            "tau":     [0.1, 0.5, 2.5, 12.5, 50.0],
            "n_ratio": [0.1, 0.25, 0.5],
        },
        "fixed": {},
        "needs_gpu": False,
    },

    "IPLSSVm": {
        # ⚠️ DEPRECADO (2026-07-10): seleção por QR-pivoting da kernel, infiel
        # ao critério α do paper. Substituído por IPLSSVmOriginal. Histórico.
        # sigma no grid: mesma razão que FSALSSVm.
        "model_name": "IPLSSVm",
        "grid": {
            "sigma":           [0.5, 1.5, 5.0],
            "tau":             [0.1, 0.5, 2.5, 12.5, 50.0],
            "selection_ratio": [0.1, 0.2, 0.5, 0.7],
        },
        "fixed": {},
        "needs_gpu": False,
    },

    "IPLSSVmOriginal": {
        # Reprodução FIEL do IP-LSSVM (Carvalho & Braga 2009): critério de
        # relevância pelo α com sinal (do LS-SVM cheio) + pseudo-inversa no
        # sistema reduzido sobredeterminado. Mesma grade do IPLSSVm.
        "model_name": "IPLSSVmOriginal",
        "grid": {
            "sigma":           [0.5, 1.5, 5.0],
            "tau":             [0.1, 0.5, 2.5, 12.5, 50.0],
            "selection_ratio": [0.1, 0.2, 0.5, 0.7],
        },
        "fixed": {},
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
        # ⚠️ DEPRECADO (2026-07-10): adaptação do projeto (k-means no espaço de
        # entrada + âncora da própria classe + fallback contra o modelo denso).
        # Substituído pelo OppositeMapsOriginalLSSVM (reprodução fiel do paper
        # Neto & Barreto 2013). Mantido aqui como histórico e ainda executável
        # via --models OppositeMapsLSSVM; fora dos runs padrão do Tier 1/2.
        # Dados históricos em results/oppmaps_adapted_history.json.
        # sigma/n_prototypes eram fixos (calibrados só nos sintéticos) —
        # última colocação em AUS/GCR/HAB/PID/VCP/AI4I no Tier 1. Revisão
        # 2026-07-07: ambos entram na grade; corrigido bug em `_train_f1`
        # (faltava multiplicar por y_sub). Explorado depois (ver memória do
        # projeto): F1 de validação em vez de treino, `val_fraction`
        # buscável — nenhuma mudou o resultado prático o bastante pra
        # justificar a complexidade, revertidas.
        # Revisão 2026-07-07 (3)-(4): tentativas de `min_sparsity`/fallback
        # contra um "piso" em vez do modelo denso — reduziram a instabilidade
        # mas pioraram F1 (ver memória do projeto) e foram revertidas.
        # Revisão 2026-07-07 (5) — causa raiz encontrada: `_select_prototypes`
        # só pegava o vizinho mais próximo da classe OPOSTA por protótipo,
        # concentrando o subconjunto inteiro em pontos de fronteira/overlap.
        # LSSVM usa mínimos quadrados (não hinge loss) — ajustar +1/-1 exatos
        # num conjunto só de pares muito próximos de classes opostas força
        # uma fronteira de decisão violentamente oscilante (kernel quase
        # saturado, F1 pior que aleatório observado no HAB). Fix: cada
        # protótipo agora também contribui uma âncora da PRÓPRIA classe, não
        # só o ponto de fronteira oposto — com isso o fallback comum (contra
        # o modelo 100% denso) voltou a ser suficiente, sem precisar de
        # `min_sparsity`: F1 ficou igual ou melhor que o modelo denso em
        # HAB/GCR/AUS/PID, com 54-64% de esparsidade real (não mais 0%).
        "model_name": "OppositeMapsLSSVM",
        "grid": {
            "sigma":        [0.05, 0.15, 0.5, 1.5, 5.0],
            "tau":          [0.1, 0.5, 2.5, 12.5],
            "n_prototypes": [5, 10, 25, 50, 100, 10_000],
        },
        "fixed": {"drop_tolerance": 0.05},
        "needs_gpu": False,
    },

    "OppositeMapsOriginalLSSVM": {
        # Reprodução FIEL do paper (Neto & Barreto 2013): Kernel K-means no
        # espaço de características (K2M), passos 3-6 do artigo, sem fallback
        # e sem âncora. Comparação lado a lado com o OppositeMapsLSSVM
        # (adaptação do projeto). Grade análoga, sem o sentinela 10_000 (era
        # do mecanismo de piso, inexistente aqui) e sem drop_tolerance.
        "model_name": "OppositeMapsOriginalLSSVM",
        "grid": {
            "sigma":       [0.05, 0.15, 0.5, 1.5, 5.0],
            "tau":         [0.1, 0.5, 2.5, 12.5],
            # protótipos por classe = fração de N (o VQ escala com o tamanho
            # do dataset). 90% de esparsidade fixa era agressivo demais; o CV
            # batia no teto 0.5, então 0.7 foi adicionado (deixa a fiel chegar
            # mais perto do denso onde compensa). Range: agressivo (0.2, ~90%
            # esparso) a suave (0.7, ~65%).
            "proto_ratio": [0.2, 0.3, 0.5, 0.7],
        },
        "fixed": {},
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
        # sigma no grid: kernel RBF colapsa com sigma=0.5 em dados UCI de alta dimensão.
        "model_name": "NystromLSSVMColnorm",
        "grid": {
            "sigma": [0.5, 1.5, 5.0],
            "gamma": [0.1, 1.0, 10.0, 30.0, 50.0, 100.0],
        },
        "fixed": {"m_ratio": 0.30},
        "needs_gpu": False,
    },

    # Baseline de ablação: idêntico ao NystromLSSVMColnorm, mas com seleção
    # de landmarks aleatória (random). Isola o efeito do seletor colnorm.
    # Ver scripts/analyze_nystrom_random.py.
    "NystromLSSVMRandom": {
        "model_name": "NystromLSSVMRandom",
        "grid": {
            "sigma": [0.5, 1.5, 5.0],
            "gamma": [0.1, 1.0, 10.0, 30.0, 50.0, 100.0],
        },
        "fixed": {"m_ratio": 0.30},
        "needs_gpu": False,
    },

    # Ablação de seletores (mesma grade do colnorm/random). K-means é o
    # baseline "inteligente" do paper ICML de Nyström; opposite = Opposite Maps.
    "NystromLSSVMKmeans": {
        "model_name": "NystromLSSVMKmeans",
        "grid": {
            "sigma": [0.5, 1.5, 5.0],
            "gamma": [0.1, 1.0, 10.0, 30.0, 50.0, 100.0],
        },
        "fixed": {"m_ratio": 0.30},
        "needs_gpu": False,
    },
    "NystromLSSVMOpposite": {
        "model_name": "NystromLSSVMOpposite",
        "grid": {
            "sigma": [0.5, 1.5, 5.0],
            "gamma": [0.1, 1.0, 10.0, 30.0, 50.0, 100.0],
        },
        "fixed": {"m_ratio": 0.30},
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
            "batch_size":     512,
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
            "batch_size":     512,
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
            "batch_size":     512,
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
            "batch_size":     512,
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
            "batch_size":         1024,  # inter-atenção B×B — maior batch = mais contexto, mas O(B²)
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
            "batch_size":         4096,  # O(B×m) — 10x menos memória que SAINT, cabe batch 4x maior
        },
        "needs_gpu": True,
    },

    # Ablação de seleção de landmarks do FT-CUR (paralela à do Nyström-SVM).
    # Reutilizam a classe FTTransformerCURColnorm; o seletor entra via `fixed`.
    # Mesma grade/fixos do colnorm; muda apenas selection_method.
    **{
        f"FTTransformerCUR{name}": {
            "model_name": "FTTransformerCURColnorm",
            "grid": {"n_layers": [1, 2, 3], "n_heads": [2, 4], "m_ratio": [0.10, 0.20]},
            "fixed": {
                "d_model": 32, "lr": 1e-3, "epochs": 40, "patience": 6,
                "early_stop_metric": "val_loss", "batch_size": 4096,
                "selection_method": method,
            },
            "needs_gpu": True,
        }
        for name, method in (("Random", "random"), ("Kmeans", "kmeans"),
                             ("Opposite", "opposite"))
    },
}


def grid_size(variant: str) -> int:
    """Number of grid points (Cartesian product) for a variant."""
    g = GRIDS[variant]["grid"]
    n = 1
    for v in g.values():
        n *= len(v)
    return n
