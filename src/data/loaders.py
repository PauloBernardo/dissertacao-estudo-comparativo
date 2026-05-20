"""Standardised dataset loaders for all 18 real datasets.

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
        """
        key = name.upper()
        if key not in cls._registry:
            available = sorted(cls._registry.keys())
            raise ValueError(f"Unknown dataset {name!r}. Available: {available}")
        X, y, meta = cls._registry[key]()
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


# ── Tier 2: Medium datasets ───────────────────────────────────────────────────

@_register("ADULT")
def _load_adult() -> tuple[NDArray, NDArray, dict]:
    """Adult Income — from sklearn fetch_openml."""
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import OrdinalEncoder

    dest_flag = _DATA_DIR / "adult_downloaded.flag"
    if not dest_flag.exists():
        logger.info("Fetching Adult dataset via openml (may take a moment)...")

    data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = data.frame.dropna()

    cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != data.target_names[0]]
    num_cols = [c for c in df.columns if c not in cat_cols and c != data.target_names[0]]

    enc = OrdinalEncoder()
    X_cat = enc.fit_transform(df[cat_cols]) if cat_cols else np.zeros((len(df), 0))
    X_num = df[num_cols].values.astype(float)
    X = np.hstack([X_num, X_cat])

    y_raw = df[data.target_names[0]].values
    y = _to_binary(y_raw, positive_value=">50K")
    dest_flag.touch()
    return X, y, {"tier": 2}


@_register("CREDIT")
def _load_credit() -> tuple[NDArray, NDArray, dict]:
    """Credit Card Default — from sklearn fetch_openml."""
    from sklearn.datasets import fetch_openml
    data = fetch_openml("default-of-credit-card-clients", version=1, as_frame=True, parser="auto")
    df = data.frame.dropna()
    target = data.target_names[0]
    X = df.drop(columns=[target]).values.astype(float)
    y_raw = df[target].values
    y = _to_binary(y_raw, positive_value="1") if y_raw.dtype == object else y_raw.astype(int)
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
    """Online Shoppers Purchasing Intention — via OpenML."""
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import OrdinalEncoder

    data = fetch_openml(data_id=42696, as_frame=True, parser="auto")
    df = data.frame.dropna()
    target = data.target_names[0]

    cat_cols = df.select_dtypes(include=["category", "object", "bool"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != target]
    num_cols = [c for c in df.columns if c not in cat_cols and c != target]

    enc = OrdinalEncoder()
    X_cat = enc.fit_transform(df[cat_cols].astype(str)) if cat_cols else np.zeros((len(df), 0))
    X_num = df[num_cols].values.astype(float)
    X = np.hstack([X_num, X_cat])

    y_raw = df[target].astype(str).values
    y = (y_raw == "True").astype(int)
    if len(np.unique(y)) != 2:
        # Fallback: map to most-common binary
        uniq = np.unique(y_raw)
        y = (y_raw == uniq[-1]).astype(int)
    return X, y, {"tier": 2}


def _load_higgs_subset(n_rows: int, tier: int) -> tuple[NDArray, NDArray, dict]:
    """Load a subset of the HIGGS dataset from OpenML (data_id=23512)."""
    dest = _DATA_DIR / "higgs_full.parquet"
    if not dest.exists():
        logger.info("Fetching HIGGS dataset (11M rows) — this may take a while...")
        from sklearn.datasets import fetch_openml
        data = fetch_openml(data_id=23512, as_frame=True, parser="auto")
        df = data.frame
        df.to_parquet(dest, index=False)
    else:
        import pandas as pd
        df = pd.read_parquet(dest)

    df = df.iloc[:n_rows]
    target = df.columns[-1]   # last column is the label
    X = df.iloc[:, :-1].values.astype(float)
    y = df[target].astype(float).astype(int).values
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
