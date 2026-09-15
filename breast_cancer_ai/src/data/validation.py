"""
Data validation module — checks dataset integrity, schema, and missing values.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_dataset(df: pd.DataFrame, name: str = "dataset") -> dict:
    """
    Run comprehensive validation checks on a DataFrame.

    Returns a validation report dict with issues, warnings, and statistics.
    """
    report = {
        "name": name,
        "shape": df.shape,
        "issues": [],
        "warnings": [],
        "stats": {},
    }

    n_rows, n_cols = df.shape

    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / n_rows * 100).round(2)
    high_missing_cols = missing_pct[missing_pct > 30].index.tolist()
    if high_missing_cols:
        report["warnings"].append(
            f"{len(high_missing_cols)} columns have >30% missing values: {high_missing_cols[:5]}..."
        )

    report["stats"]["missing_values"] = {
        "total_missing": int(missing.sum()),
        "pct_missing": round(missing.sum() / (n_rows * n_cols) * 100, 2),
        "cols_with_missing": int((missing > 0).sum()),
        "cols_high_missing": len(high_missing_cols),
    }

    # Duplicate rows (fast index check if high-dimensional)
    if n_cols > 1000:
        n_dupes = int(df.index.duplicated().sum())
    else:
        n_dupes = int(df.duplicated().sum())
    if n_dupes > 0:
        report["warnings"].append(f"{n_dupes} duplicate rows found.")

    # Constant columns (no variance) using fast numpy
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        vals = df[numeric_cols].values
        stds = np.nanstd(vals, axis=0)
        zero_std_idx = np.where(stds == 0)[0]
        const_cols = [numeric_cols[i] for i in zero_std_idx]
        if const_cols:
            report["warnings"].append(
                f"{len(const_cols)} constant columns (zero variance): {const_cols[:5]}..."
            )
        report["stats"]["constant_columns"] = len(const_cols)

    # Data types
    report["stats"]["dtypes"] = {
        "numeric": int(len(numeric_cols)),
        "categorical": int(len(df.columns) - len(numeric_cols)),
        "boolean": 0,
    }

    # Extreme values / potential outliers (fast numpy check)
    if len(numeric_cols) > 0:
        check_cols = numeric_cols if len(numeric_cols) <= 1000 else numeric_cols[:1000]
        sub_vals = df[check_cols].values
        m = np.nanmean(sub_vals, axis=0)
        s = np.nanstd(sub_vals, axis=0) + 1e-8
        z = np.abs((sub_vals - m) / s)
        outlier_count = int((z > 5).sum())
        if outlier_count > 0:
            report["warnings"].append(
                f"{outlier_count} extreme outlier values detected (|z| > 5)."
            )
        report["stats"]["outliers"] = outlier_count

    if not report["issues"]:
        logger.info(f"[{name}] Validation passed with {len(report['warnings'])} warnings.")
    else:
        logger.warning(f"[{name}] Validation found {len(report['issues'])} issues.")

    return report


def validate_target(df: pd.DataFrame, target_col: str, name: str = "dataset") -> dict:
    """Validate the target column (pCR label)."""
    report = {"name": name, "target_col": target_col, "issues": [], "warnings": []}

    if target_col not in df.columns:
        report["issues"].append(f"Target column '{target_col}' not found.")
        return report

    target = df[target_col]
    n_missing = target.isnull().sum()
    value_counts = target.value_counts()

    report["stats"] = {
        "n_missing": int(n_missing),
        "value_counts": value_counts.to_dict(),
        "class_balance_ratio": round(
            float(value_counts.min() / value_counts.max()), 3
        ) if len(value_counts) > 1 else 1.0,
    }

    if n_missing > 0:
        report["warnings"].append(f"Target has {n_missing} missing values.")

    # Check binary
    unique_vals = sorted(target.dropna().unique().tolist())
    if unique_vals not in [[0, 1], [0.0, 1.0]]:
        report["warnings"].append(
            f"Target values are not binary 0/1: {unique_vals}"
        )

    # Class imbalance
    ratio = report["stats"]["class_balance_ratio"]
    if ratio < 0.3:
        report["warnings"].append(
            f"Severe class imbalance detected (ratio={ratio}). Consider class weighting."
        )

    return report


def validate_alignment(dataframes: dict[str, pd.DataFrame]) -> dict:
    """
    Check that all modalities share the same patient IDs.
    """
    report = {"aligned": True, "issues": [], "common_patients": 0}

    non_empty = {k: v for k, v in dataframes.items() if not v.empty}
    if len(non_empty) < 2:
        return report

    index_sets = {k: set(v.index) for k, v in non_empty.items()}
    all_patients = set.union(*index_sets.values())
    common_patients = set.intersection(*index_sets.values())

    report["common_patients"] = len(common_patients)
    report["all_patients"] = len(all_patients)
    report["missing_per_modality"] = {
        k: len(all_patients - s) for k, s in index_sets.items()
    }

    if len(common_patients) < len(all_patients):
        report["issues"].append(
            f"Only {len(common_patients)}/{len(all_patients)} patients present in all modalities."
        )

    return report
