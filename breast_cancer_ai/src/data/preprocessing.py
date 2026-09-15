"""
Data preprocessing — normalization, imputation, encoding, and splitting.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────

def normalize(
    df: pd.DataFrame,
    method: str = "zscore",
    fitted_scaler=None,
) -> tuple[pd.DataFrame, object]:
    """
    Normalize numeric data.

    Args:
        df: Input DataFrame (numeric)
        method: 'zscore' | 'minmax' | 'log1p'
        fitted_scaler: Pre-fitted scaler (use for val/test sets)

    Returns:
        (normalized_df, scaler)
    """
    if method == "log1p":
        # Apply log1p transform in-place, no scaler needed
        df_out = np.log1p(df.clip(lower=0))
        return df_out, None

    ScalerClass = StandardScaler if method == "zscore" else MinMaxScaler
    scaler = fitted_scaler or ScalerClass()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_out = df.copy()

    if fitted_scaler is None:
        df_out[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df_out[numeric_cols] = scaler.transform(df[numeric_cols])

    # Zero-variance columns produce NaN after z-score (std=0 → 0/0). Fill with 0.
    nan_count = int(df_out[numeric_cols].isna().sum().sum())
    if nan_count > 0:
        logger.warning(f"normalize: {nan_count} NaN after {method} (zero-var cols). Filling with 0.")
        df_out[numeric_cols] = df_out[numeric_cols].fillna(0.0)

    logger.debug(f"Normalized {len(numeric_cols)} numeric columns using {method}.")
    return df_out, scaler


# ─────────────────────────────────────────────
# Imputation
# ─────────────────────────────────────────────

def impute(
    df: pd.DataFrame,
    method: str = "median",
    fitted_imputer=None,
    n_neighbors: int = 5,
) -> tuple[pd.DataFrame, object]:
    """
    Impute missing values.

    Args:
        method: 'mean' | 'median' | 'mode' | 'knn' | 'drop'
        fitted_imputer: Pre-fitted imputer for val/test
    """
    if method == "drop":
        df_out = df.dropna()
        return df_out, None

    if method == "knn":
        imputer = fitted_imputer or KNNImputer(n_neighbors=n_neighbors)
    else:
        strategy = "most_frequent" if method == "mode" else method
        imputer = fitted_imputer or SimpleImputer(strategy=strategy)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_out = df.copy()

    if len(numeric_cols) > 0:
        if fitted_imputer is None:
            df_out[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        else:
            df_out[numeric_cols] = imputer.transform(df[numeric_cols])

    n_missing_before = df[numeric_cols].isnull().sum().sum()
    n_missing_after = df_out[numeric_cols].isnull().sum().sum()
    logger.debug(
        f"Imputed {n_missing_before - n_missing_after} missing values using {method}."
    )
    return df_out, imputer


# ─────────────────────────────────────────────
# Encoding
# ─────────────────────────────────────────────

def encode_categorical(
    df: pd.DataFrame,
    method: str = "onehot",
    fitted_encoder=None,
) -> tuple[pd.DataFrame, object]:
    """
    Encode categorical columns.

    Args:
        method: 'onehot' | 'label' | 'ordinal'
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        return df, None

    if method == "onehot":
        encoder = fitted_encoder or OneHotEncoder(
            sparse_output=False, handle_unknown="ignore", drop="if_binary"
        )
        if fitted_encoder is None:
            encoded = encoder.fit_transform(df[cat_cols])
        else:
            encoded = encoder.transform(df[cat_cols])

        encoded_cols = encoder.get_feature_names_out(cat_cols)
        encoded_df = pd.DataFrame(encoded, index=df.index, columns=encoded_cols)
        df_out = pd.concat([df.drop(columns=cat_cols), encoded_df], axis=1)
    else:
        encoder = {}
        df_out = df.copy()
        for col in cat_cols:
            le = fitted_encoder.get(col, LabelEncoder()) if fitted_encoder else LabelEncoder()
            if fitted_encoder is None:
                df_out[col] = le.fit_transform(df[col].astype(str))
            else:
                df_out[col] = le.transform(df[col].astype(str))
            encoder[col] = le

    logger.debug(f"Encoded {len(cat_cols)} categorical columns using {method}.")
    return df_out, encoder


# ─────────────────────────────────────────────
# Feature filtering
# ─────────────────────────────────────────────

def filter_low_variance(
    df: pd.DataFrame,
    threshold: float = 0.01,
) -> tuple[pd.DataFrame, list[str]]:
    """Remove near-zero variance features."""
    numeric_df = df.select_dtypes(include=[np.number])
    variances = numeric_df.var()
    keep_cols = variances[variances > threshold].index.tolist()
    removed = [c for c in numeric_df.columns if c not in keep_cols]
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    df_out = df[keep_cols + non_numeric]
    logger.info(f"Variance filter: kept {len(keep_cols)}, removed {len(removed)} features.")
    return df_out, removed


def filter_high_missing(
    df: pd.DataFrame,
    threshold: float = 0.3,
) -> tuple[pd.DataFrame, list[str]]:
    """Remove columns with more than `threshold` fraction of missing values."""
    missing_frac = df.isnull().mean()
    drop_cols = missing_frac[missing_frac > threshold].index.tolist()
    df_out = df.drop(columns=drop_cols)
    logger.info(
        f"Missing filter (>{threshold*100:.0f}%): removed {len(drop_cols)} columns."
    )
    return df_out, drop_cols


# ─────────────────────────────────────────────
# Dataset splitting
# ─────────────────────────────────────────────

def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> tuple:
    """
    Patient-level stratified split into train / validation / test.

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: train+val vs test
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_val_idx, test_idx = next(sss.split(X, y))
    X_trainval = X.iloc[train_val_idx]
    y_trainval = y.iloc[train_val_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    # Second split: train vs val
    val_frac = val_size / (1 - test_size)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_frac, random_state=random_state
    )
    train_idx, val_idx = next(sss2.split(X_trainval, y_trainval))
    X_train = X_trainval.iloc[train_idx]
    y_train = y_trainval.iloc[train_idx]
    X_val = X_trainval.iloc[val_idx]
    y_val = y_trainval.iloc[val_idx]

    logger.info(
        f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)} | "
        f"pCR rates: train={y_train.mean():.2%}, val={y_val.mean():.2%}, test={y_test.mean():.2%}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
