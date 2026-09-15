"""
Patient-level alignment of multi-omics modalities.
Ensures all modalities share the same patients and consistent ordering.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def align_modalities(
    dataframes: dict[str, pd.DataFrame],
    strategy: str = "inner",
) -> dict[str, pd.DataFrame]:
    """
    Align multiple omics DataFrames to a common patient set.

    Args:
        dataframes: Dict of modality name -> DataFrame (indexed by patient_id)
        strategy: 'inner' (only common patients) or 'outer' (all patients, fill NaN)

    Returns:
        Dict with same keys, all DataFrames sharing the same index.
    """
    non_empty = {k: v for k, v in dataframes.items() if v is not None and not v.empty}
    if not non_empty:
        logger.warning("No non-empty DataFrames to align.")
        return dataframes

    # Determine common patient set
    index_sets = [set(df.index) for df in non_empty.values()]
    if strategy == "inner":
        common_ids = sorted(set.intersection(*index_sets))
        logger.info(
            f"Alignment (inner): {len(common_ids)} common patients across "
            f"{list(non_empty.keys())}"
        )
    else:
        common_ids = sorted(set.union(*index_sets))
        logger.info(
            f"Alignment (outer): {len(common_ids)} total patients across "
            f"{list(non_empty.keys())}"
        )

    if len(common_ids) == 0:
        logger.error("No overlapping patient IDs found across modalities!")
        return dataframes

    aligned: dict[str, pd.DataFrame] = {}
    for key, df in dataframes.items():
        if df is None or df.empty:
            aligned[key] = df
            continue
        aligned[key] = df.reindex(common_ids)
        n_missing = aligned[key].isnull().any(axis=1).sum()
        logger.debug(f"  [{key}] {aligned[key].shape} | {n_missing} patients with NaN")

    return aligned


def align_with_target(
    modalities: dict[str, pd.DataFrame],
    target: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """
    Align all modality DataFrames with the target (pCR) labels.
    Drops patients missing the target label.

    Returns:
        aligned_modalities: Dict of aligned modality DataFrames
        y: Aligned target Series
    """
    target_col = target.columns[0] if hasattr(target, "columns") else "pCR"
    y = target[target_col].dropna()
    valid_ids = sorted(y.index.tolist())

    aligned = {}
    for key, df in modalities.items():
        if df is None or df.empty:
            aligned[key] = df
            continue
        shared = [pid for pid in valid_ids if pid in df.index]
        aligned[key] = df.reindex(shared)

    y = y.reindex(valid_ids)
    logger.info(
        f"After target alignment: {len(valid_ids)} patients with labels "
        f"(pCR=1: {int(y.sum())}, pCR=0: {int((y==0).sum())})"
    )
    return aligned, y


def check_id_consistency(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    name1: str = "df1",
    name2: str = "df2",
) -> dict:
    """Check patient ID consistency between two DataFrames."""
    ids1 = set(df1.index)
    ids2 = set(df2.index)
    return {
        "common": len(ids1 & ids2),
        "only_in_first": len(ids1 - ids2),
        "only_in_second": len(ids2 - ids1),
        "union": len(ids1 | ids2),
    }
