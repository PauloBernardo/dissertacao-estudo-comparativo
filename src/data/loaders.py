"""Standardised dataset loaders for all 19 real datasets.

Usage:
    X, y, meta = DatasetLoader.load("BCW")
    X, y, meta = DatasetLoader.load("HAB")

Returns
-------
X   : np.ndarray, shape (n_samples, n_features)  — raw, unscaled features
y   : np.ndarray, shape (n_samples,)              — binary labels in {0, 1}
meta: dict with keys: name, n_samples, n_features, tier, class_ratio
"""

from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Data cache directory
_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
_DATA_DIR.mkdir(parents=True, exist_ok=True)


class DatasetLoader:
    """Factory class for loading datasets by short name."""

    _registry: dict[str, "_LoaderFunc"] = {}

    @classmethod
    def load(cls, name: str) -> tuple[NDArray, NDArray, dict]:
        """Load a dataset by its short name (e.g. 'BCW', 'HAB').

        Returns (X, y, meta) where y ∈ {0, 1} (positive class = 1).
        Checks for a cached .parquet in data/raw/ before downloading.
        """
        key = name.upper()
        if key not in cls._registry:
            available = sorted(cls._registry.keys())
            raise ValueError(f"Unknown dataset {name!r}. Available: {available}")

        cache = _DATA_DIR / f"{key}.parquet"
        if cache.exists():
            df = pd.read_parquet(cache)
            y   = df.pop("__y__").values.astype(int)
            X   = df.values.astype(float)
            meta = {"tier": df.attrs.get("__tier__")}
        else:
            X, y, meta = cls._registry[key]()
            df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
            df["__y__"] = y.astype(int)
            df.attrs["__tier__"] = meta.get("tier")
            df.to_parquet(cache, index=False)
            logger.info("Cached %s → %s", key, cache)

        meta["name"] = key
        meta.setdefault("n_samples", len(y))
        meta.setdefault("n_features", X.shape[1])
        meta.setdefault("class_ratio", float(np.mean(y)))
        return X, y, meta

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._registry.keys())


def _register(name: str):
    """Decorator to register a loader function."""
    def decorator(fn):
        DatasetLoader._registry[name.upper()] = fn
        return fn
    return decorator


# ── Utilities ─────────────────────────────────────────────────────────────────

def _download(url: str, dest: Path) -> None:
    """Download a file if it doesn't already exist."""
    if dest.exists():
        return
    logger.info("Downloading %s → %s", url, dest)
    urllib.request.urlretrieve(url, dest)


def _to_binary(y_raw: NDArray, positive_value) -> NDArray:
    """Convert raw labels to {0, 1}."""
    y = np.where(y_raw == positive_value, 1, 0).astype(int)
    if len(np.unique(y)) != 2:
        raise ValueError(f"Expected 2 classes after mapping, got {np.unique(y)}.")
    return y


# ── Tier 1: Small datasets ────────────────────────────────────────────────────

@_register("BCW")
def _load_bcw() -> tuple[NDArray, NDArray, dict]:
    """Breast Cancer Wisconsin — from sklearn."""
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = data.data.astype(float)
    y = data.target.astype(int)          # 1 = malignant (positive)
    return X, y, {"tier": 1}


@_register("PID")
def _load_pid() -> tuple[NDArray, NDArray, dict]:
    """Pima Indians Diabetes — downloaded from GitHub mirror."""
    dest = _DATA_DIR / "pima.csv"
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    _download(url, dest)
    df = pd.read_csv(dest, header=None)
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values.astype(int)
    return X, y, {"tier": 1}


@_register("HAB")
def _load_hab() -> tuple[NDArray, NDArray, dict]:
    """Haberman Survival — from UCI."""
    dest = _DATA_DIR / "haberman.data"
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"
    _download(url, dest)
    df = pd.read_csv(dest, header=None)
    X = df.iloc[:, :-1].values.astype(float)
    y_raw = df.iloc[:, -1].values  # 1=survived 5 years, 2=did not
    y = _to_binary(y_raw, positive_value=1)
    return X, y, {"tier": 1}


@_register("VCP")
def _load_vcp() -> tuple[NDArray, NDArray, dict]:
    """Vertebral Column (2-class: AB vs NO) — via sklearn/openml."""
    from sklearn.datasets import fetch_openml
    data = fetch_openml(data_id=1523, as_frame=True, parser="auto")  # vertebra-column
    df = data.frame.dropna()
    target = data.target_names[0] if hasattr(data, "target_names") else "Class"
    X = df.drop(columns=[target]).values.astype(float)
    y_raw = df[target].astype(str).values
    # OpenML id=1523: class '2' = Normal → 0; '1' and '3' = Abnormal → 1
    y = np.where(y_raw == "2", 0, 1).astype(int)
    return X, y, {"tier": 1}


@_register("GCR")
def _load_gcr() -> tuple[NDArray, NDArray, dict]:
    """German Credit — from UCI (numeric version)."""
    dest = _DATA_DIR / "german.data-numeric"
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric"
    _download(url, dest)
    df = pd.read_csv(dest, sep=r"\s+", header=None)
    X = df.iloc[:, :-1].values.astype(float)
    y_raw = df.iloc[:, -1].values   # 1=good, 2=bad
    y = _to_binary(y_raw, positive_value=1)
    return X, y, {"tier": 1}


@_register("AUS")
def _load_aus() -> tuple[NDArray, NDArray, dict]:
    """Australian Credit Approval — from UCI."""
    dest = _DATA_DIR / "australian.dat"
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat"
    _download(url, dest)
    df = pd.read_csv(dest, sep=" ", header=None)
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values.astype(int)   # 0 or 1
    return X, y, {"tier": 1}


# ── Tier 1 additions ──────────────────────────────────────────────────────────

_AI4I_SEED = 42


def _load_ai4i_features() -> tuple[pd.DataFrame, NDArray]:
    """Download and parse the AI4I 2020 raw features and label.

    Drops `UDI`/`Product ID` (identifiers) and `TWF`/`HDF`/`PWF`/`OSF`/`RNF`:
    those five failure-mode flags agree with `Machine failure` in ~99.7% of
    rows (they are literally the label's own decomposition), so keeping them
    as features would leak the target almost perfectly.
    """
    dest_zip = _DATA_DIR / "ai4i2020.zip"
    dest_csv = _DATA_DIR / "ai4i2020.csv"
    if not dest_csv.exists():
        _download(
            "https://archive.ics.uci.edu/static/public/601/"
            "ai4i+2020+predictive+maintenance+dataset.zip",
            dest_zip,
        )
        with zipfile.ZipFile(dest_zip, "r") as z:
            with z.open("ai4i2020.csv") as f:
                dest_csv.write_bytes(f.read())

    df = pd.read_csv(dest_csv)
    leak_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    df = df.drop(columns=["UDI", "Product ID"] + leak_cols)
    y = df.pop("Machine failure").astype(int).values
    return df, y


@_register("AI4I")
def _load_ai4i() -> tuple[NDArray, NDArray, dict]:
    """AI4I 2020 Predictive Maintenance — undersampled 1:3, placed in Tier 1.

    Original dataset: 10 000 samples, 3.39% failures (339/10000). Applies a
    deterministic (seed=42) undersampling of the majority class at a 1:3
    ratio (minority:majority), keeping all 339 failure cases and yielding
    1356 samples — this is the resampled form used for the benchmark, not
    the raw 10k dataset.
    """
    from sklearn.preprocessing import OrdinalEncoder

    df, y_full = _load_ai4i_features()

    cat_cols = ["Type"]
    num_cols = [c for c in df.columns if c not in cat_cols]
    enc = OrdinalEncoder()
    X_cat = enc.fit_transform(df[cat_cols])
    X_num = df[num_cols].values.astype(float)
    X_full = np.hstack([X_num, X_cat])

    rng = np.random.RandomState(_AI4I_SEED)
    idx_pos = np.where(y_full == 1)[0]
    idx_neg = rng.choice(np.where(y_full == 0)[0], size=3 * len(idx_pos), replace=False)
    idx = np.sort(np.concatenate([idx_pos, idx_neg]))

    return X_full[idx], y_full[idx], {"tier": 1}


# ── Tier 2: Medium datasets ───────────────────────────────────────────────────

@_register("ADULT")
def _load_adult() -> tuple[NDArray, NDArray, dict]:
    """Adult Income — UCI direct download (train + test combined)."""
    from sklearn.preprocessing import OrdinalEncoder

    dest_train = _DATA_DIR / "adult.data"
    dest_test  = _DATA_DIR / "adult.test"
    _download("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
              dest_train)
    _download("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
              dest_test)

    cols = ["age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
    df_train = pd.read_csv(dest_train, header=None, names=cols,
                           skipinitialspace=True, na_values="?")
    df_test  = pd.read_csv(dest_test,  header=None, names=cols,
                           skipinitialspace=True, na_values="?", skiprows=1)
    df = pd.concat([df_train, df_test], ignore_index=True).dropna()
    df["income"] = df["income"].str.rstrip(".")  # test set has trailing dot

    target = "income"
    cat_cols = ["workclass", "education", "marital-status", "occupation",
                "relationship", "race", "sex", "native-country"]
    num_cols = ["age", "fnlwgt", "education-num", "capital-gain",
                "capital-loss", "hours-per-week"]

    enc = OrdinalEncoder()
    X_cat = enc.fit_transform(df[cat_cols])
    X_num = df[num_cols].values.astype(float)
    X = np.hstack([X_num, X_cat])
    y = (df[target] == ">50K").astype(int).values
    return X, y, {"tier": 2}


@_register("CREDIT")
def _load_credit() -> tuple[NDArray, NDArray, dict]:
    """Credit Card Default — UCI direct download (Excel)."""
    dest = _DATA_DIR / "credit_default.xls"
    _download(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/"
        "default%20of%20credit%20card%20clients.xls",
        dest,
    )
    df = pd.read_excel(dest, header=1)  # row 0 is a secondary header
    target = "default payment next month"
    X = df.drop(columns=["ID", target], errors="ignore").values.astype(float)
    y = df[target].astype(int).values
    return X, y, {"tier": 2}


# ── Tier 2: Additional medium datasets ────────────────────────────────────────

@_register("BANK")
def _load_bank() -> tuple[NDArray, NDArray, dict]:
    """Bank Marketing — from UCI (full, semicolon-separated)."""
    from sklearn.preprocessing import OrdinalEncoder

    dest = _DATA_DIR / "bank_full.csv"
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
    zip_dest = _DATA_DIR / "bank.zip"
    if not dest.exists():
        _download(url, zip_dest)
        with zipfile.ZipFile(zip_dest, "r") as z:
            # The zip contains bank-full.csv
            names = z.namelist()
            full_name = next((n for n in names if "full" in n), names[0])
            with z.open(full_name) as f:
                content = f.read().decode("utf-8")
            dest.write_text(content)

    df = pd.read_csv(dest, sep=";")
    target = "y"
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != target]
    num_cols = [c for c in df.columns if c not in cat_cols and c != target]

    enc = OrdinalEncoder()
    X_cat = enc.fit_transform(df[cat_cols]) if cat_cols else np.zeros((len(df), 0))
    X_num = df[num_cols].values.astype(float)
    X = np.hstack([X_num, X_cat])
    y = (df[target] == "yes").astype(int).values
    return X, y, {"tier": 2}


@_register("TELCO")
def _load_telco() -> tuple[NDArray, NDArray, dict]:
    """IBM Telco Churn — from GitHub CSV mirror."""
    from sklearn.preprocessing import OrdinalEncoder

    dest = _DATA_DIR / "telco_churn.csv"
    url = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
           "/master/data/Telco-Customer-Churn.csv")
    _download(url, dest)

    df = pd.read_csv(dest)
    # Drop customer ID, handle TotalCharges (may be whitespace)
    df = df.drop(columns=["customerID"], errors="ignore")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    target = "Churn"
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != target]
    num_cols = [c for c in df.columns if c not in cat_cols and c != target]

    enc = OrdinalEncoder()
    X_cat = enc.fit_transform(df[cat_cols]) if cat_cols else np.zeros((len(df), 0))
    X_num = df[num_cols].values.astype(float)
    X = np.hstack([X_num, X_cat])
    y = (df[target] == "Yes").astype(int).values
    return X, y, {"tier": 2}


@_register("SHOPPERS")
def _load_shoppers() -> tuple[NDArray, NDArray, dict]:
    """Online Shoppers Purchasing Intention — UCI direct download."""
    from sklearn.preprocessing import OrdinalEncoder

    dest = _DATA_DIR / "online_shoppers_intention.csv"
    _download(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/"
        "online_shoppers_intention.csv",
        dest,
    )
    df = pd.read_csv(dest).dropna()
    target = "Revenue"

    cat_cols = ["Month", "VisitorType", "Weekend"]
    num_cols = [c for c in df.columns if c not in cat_cols and c != target]

    enc = OrdinalEncoder()
    X_cat = enc.fit_transform(df[cat_cols].astype(str))
    X_num = df[num_cols].values.astype(float)
    X = np.hstack([X_num, X_cat])
    y = df[target].astype(int).values
    return X, y, {"tier": 2}


# ── Tier 2 "fully balanced" datasets (CREDIT_BAL, BANK_BAL, SHOPPERS_BAL) ─────
#
# ⚠️ ATENÇÃO METODOLÓGICA (revisão 29/05/2026):
#
# Esses loaders balanceiam o dataset ANTES do split train/test. Isto contamina
# o conjunto de teste, tornando-o diferente da distribuição original. Para
# testar a hipótese H1 "balanced training fixes the collapse" de forma justa,
# use o protocolo correto via:
#
#     scripts/run_tier2_balanced.py --group <group>
#
# que aplica o balanceamento APENAS no treino após o split, mantendo o teste
# com a distribuição original (imbalanceada). Estes loaders foram mantidos
# como referência caso seja necessário um benchmark 50/50 de ponta a ponta,
# mas NÃO devem ser usados para testar H1.
#
# Implementação: undersampling determinístico (seed=42) da classe majoritária.

_BAL_SEED = 42


def _balance_undersample(X: np.ndarray, y: np.ndarray,
                          seed: int = _BAL_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Random undersampling da classe majoritária para igualar a minoritária.

    Sem dados sintéticos. Resultado: |classe 0| == |classe 1|.
    """
    rng = np.random.RandomState(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    n_minor = min(len(idx_pos), len(idx_neg))
    if len(idx_pos) > n_minor:
        idx_pos = rng.choice(idx_pos, size=n_minor, replace=False)
    if len(idx_neg) > n_minor:
        idx_neg = rng.choice(idx_neg, size=n_minor, replace=False)
    idx = np.concatenate([idx_pos, idx_neg])
    idx = np.sort(idx)  # mantém ordem original (mais determinístico)
    return X[idx], y[idx]


def _warn_fully_balanced(name: str, original: str) -> None:
    """Emite warning runtime indicando que esse loader balanceia o teste também."""
    import warnings
    warnings.warn(
        f"\n{'=' * 70}\n"
        f"⚠️  Loader {name} balanceia o dataset ANTES do split → teste também\n"
        f"   fica 50/50, comparação com Tier 2 imbalanceado fica contaminada.\n"
        f"\n"
        f"   Para H1 (testar se balanceamento corrige colapso), use:\n"
        f"     scripts/run_tier2_balanced.py --group <group>\n"
        f"   que usa o dataset {original} original + balanceamento\n"
        f"   APENAS no treino via flag --balance-train do runner.\n"
        f"{'=' * 70}",
        stacklevel=3,
    )


@_register("CREDIT_BAL")
def _load_credit_bal() -> tuple[NDArray, NDArray, dict]:
    """CREDIT 50/50 via undersampling. ⚠️ Use run_tier2_balanced.py para H1."""
    _warn_fully_balanced("CREDIT_BAL", "CREDIT")
    X, y, _ = _load_credit()
    X_b, y_b = _balance_undersample(X, y)
    return X_b, y_b, {"tier": 2, "balanced": True, "original": "CREDIT"}


@_register("BANK_BAL")
def _load_bank_bal() -> tuple[NDArray, NDArray, dict]:
    """BANK 50/50 via undersampling. ⚠️ Use run_tier2_balanced.py para H1."""
    _warn_fully_balanced("BANK_BAL", "BANK")
    X, y, _ = _load_bank()
    X_b, y_b = _balance_undersample(X, y)
    return X_b, y_b, {"tier": 2, "balanced": True, "original": "BANK"}


@_register("SHOPPERS_BAL")
def _load_shoppers_bal() -> tuple[NDArray, NDArray, dict]:
    """SHOPPERS 50/50 via undersampling. ⚠️ Use run_tier2_balanced.py para H1."""
    _warn_fully_balanced("SHOPPERS_BAL", "SHOPPERS")
    X, y, _ = _load_shoppers()
    X_b, y_b = _balance_undersample(X, y)
    return X_b, y_b, {"tier": 2, "balanced": True, "original": "SHOPPERS"}


def _load_higgs_subset(n_rows: int, tier: int) -> tuple[NDArray, NDArray, dict]:
    """Load a subset of the HIGGS dataset — streamed from UCI gz file."""
    dest_parquet = _DATA_DIR / f"higgs_{n_rows}.parquet"
    if dest_parquet.exists():
        df = pd.read_parquet(dest_parquet)
    else:
        dest_gz = _DATA_DIR / "HIGGS.csv.gz"
        if not dest_gz.exists():
            logger.info("Downloading HIGGS.csv.gz (~2.6 GB) — this may take a while...")
            _download(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz",
                dest_gz,
            )
        logger.info("Reading first %d rows from HIGGS.csv.gz...", n_rows)
        col_names = ["label"] + [f"f{i}" for i in range(1, 29)]
        df = pd.read_csv(dest_gz, header=None, names=col_names, nrows=n_rows)
        df.to_parquet(dest_parquet, index=False)

    X = df.iloc[:, 1:].values.astype(float)
    y = df.iloc[:, 0].astype(int).values
    return X, y, {"tier": tier}


@_register("HIGGS50K")
def _load_higgs50k() -> tuple[NDArray, NDArray, dict]:
    """HIGGS Boson — 50 000-sample subset (Baldi et al., 2014)."""
    return _load_higgs_subset(50_000, tier=2)


@_register("HIGGS500K")
def _load_higgs500k() -> tuple[NDArray, NDArray, dict]:
    """HIGGS Boson — 500 000-sample subset."""
    return _load_higgs_subset(500_000, tier=3)


# ── Tier 3 (sklearn built-ins) ───────────────────────────────────────────────

@_register("COVER")
def _load_cover() -> tuple[NDArray, NDArray, dict]:
    """Covertype (class 1 vs 2 binary subset) — from sklearn."""
    from sklearn.datasets import fetch_covtype
    data = fetch_covtype()
    mask = np.isin(data.target, [1, 2])
    X = data.data[mask].astype(float)
    y = (data.target[mask] == 1).astype(int)
    return X, y, {"tier": 3}


@_register("KDD99")
def _load_kdd99() -> tuple[NDArray, NDArray, dict]:
    """KDD Cup 99 (normal vs attack, 500k subset) — from sklearn."""
    from sklearn.datasets import fetch_kddcup99
    from sklearn.preprocessing import OrdinalEncoder

    data = fetch_kddcup99(subset=None, as_frame=True, percent10=True)
    df = data.frame

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    target_col = data.target_names[0] if hasattr(data, "target_names") else "labels"
    cat_cols = [c for c in cat_cols if c != target_col]

    enc = OrdinalEncoder()
    X_cat = enc.fit_transform(df[cat_cols]) if cat_cols else np.zeros((len(df), 0))
    num_cols = [c for c in df.columns if c not in cat_cols and c != target_col]
    X_num = df[num_cols].values.astype(float)
    X = np.hstack([X_num, X_cat])

    y_raw = df[target_col].values
    y = (y_raw == b"normal.").astype(int)
    return X, y, {"tier": 3}


# ── Synthetic ─────────────────────────────────────────────────────────────────

@_register("TWS")
def _load_tws() -> tuple[NDArray, NDArray, dict]:
    from src.data.synthetic import make_tws
    X, y = make_tws(n_samples=400)
    return X, (y + 1) // 2, {"tier": 0}   # convert {-1,1} → {0,1}


@_register("TWM")
def _load_twm() -> tuple[NDArray, NDArray, dict]:
    from src.data.synthetic import make_twm
    X, y = make_twm(n_samples=400)
    return X, (y + 1) // 2, {"tier": 0}


@_register("TWC")
def _load_twc() -> tuple[NDArray, NDArray, dict]:
    from src.data.synthetic import make_twc
    X, y = make_twc(n_samples=400)
    return X, (y + 1) // 2, {"tier": 0}


# ── MK5 series — 5 informative features, make_classification ──────────────────

@_register("MKE")
def _load_mke() -> tuple[NDArray, NDArray, dict]:
    from src.data.synthetic import make_mke
    X, y = make_mke(n_samples=400)
    return X, (y + 1) // 2, {"tier": 0}


@_register("MKM")
def _load_mkm() -> tuple[NDArray, NDArray, dict]:
    from src.data.synthetic import make_mkm
    X, y = make_mkm(n_samples=400)
    return X, (y + 1) // 2, {"tier": 0}


@_register("MKH")
def _load_mkh() -> tuple[NDArray, NDArray, dict]:
    from src.data.synthetic import make_mkh
    X, y = make_mkh(n_samples=400)
    return X, (y + 1) // 2, {"tier": 0}


# ── Synthetic 5-feature (N=400, 2D structure + 3 noise features) ──────────────

@_register("TWS_5f")
def _load_tws_5f() -> tuple[NDArray, NDArray, dict]:
    from src.data.synthetic import make_tws_5f
    X, y = make_tws_5f(n_samples=400)
    return X, (y + 1) // 2, {"tier": 0}


@_register("TWM_5f")
def _load_twm_5f() -> tuple[NDArray, NDArray, dict]:
    from src.data.synthetic import make_twm_5f
    X, y = make_twm_5f(n_samples=400)
    return X, (y + 1) // 2, {"tier": 0}


@_register("TWC_5f")
def _load_twc_5f() -> tuple[NDArray, NDArray, dict]:
    from src.data.synthetic import make_twc_5f
    X, y = make_twc_5f(n_samples=400)
    return X, (y + 1) // 2, {"tier": 0}


# ── Synthetic scaled (N=2000) ──────────────────────────────────────────────────

@_register("TWS_2k")
def _load_tws_2k() -> tuple[NDArray, NDArray, dict]:
    from src.data.synthetic import make_tws
    X, y = make_tws(n_samples=2000)
    return X, (y + 1) // 2, {"tier": 0}


@_register("TWM_2k")
def _load_twm_2k() -> tuple[NDArray, NDArray, dict]:
    from src.data.synthetic import make_twm
    X, y = make_twm(n_samples=2000)
    return X, (y + 1) // 2, {"tier": 0}


@_register("TWC_2k")
def _load_twc_2k() -> tuple[NDArray, NDArray, dict]:
    from src.data.synthetic import make_twc
    X, y = make_twc(n_samples=2000)
    return X, (y + 1) // 2, {"tier": 0}
