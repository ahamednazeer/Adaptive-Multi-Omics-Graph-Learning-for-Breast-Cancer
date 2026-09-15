"""
Feature selection — variance/expression filtering.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def filter_by_variance(
    df: pd.DataFrame,
    threshold: float = 0.01,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove features with variance below threshold.

    Args:
        df: Numeric DataFrame (rows=patients, cols=features)
        threshold: Minimum variance to keep a feature

    Returns:
        (filtered_df, removed_feature_names)
    """
    variances = df.var()
    keep = variances[variances >= threshold].index.tolist()
    removed = variances[variances < threshold].index.tolist()
    logger.info(
        f"Variance filter (threshold={threshold}): "
        f"kept {len(keep)}, removed {len(removed)} features"
    )
    return df[keep], removed


def filter_by_expression(
    rna_df: pd.DataFrame,
    min_mean_expression: float = 1.0,
    min_expressed_fraction: float = 0.1,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Filter low-expression RNA genes.

    Args:
        rna_df: RNA expression DataFrame
        min_mean_expression: Minimum mean expression across all patients
        min_expressed_fraction: Minimum fraction of patients with expression > 0

    Returns:
        (filtered_df, removed_genes)
    """
    mean_expr = rna_df.mean()
    expressed_frac = (rna_df > 0).mean()

    keep = mean_expr.index[
        (mean_expr >= min_mean_expression) & (expressed_frac >= min_expressed_fraction)
    ].tolist()
    removed = [c for c in rna_df.columns if c not in keep]

    logger.info(
        f"Expression filter: kept {len(keep)}, removed {len(removed)} genes "
        f"(min_mean={min_mean_expression}, min_expressed={min_expressed_fraction:.0%})"
    )
    return rna_df[keep], removed


def filter_protein_quality(
    protein_df: pd.DataFrame,
    missing_threshold: float = 0.3,
    cv_threshold: float = 0.05,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Filter low-quality protein features.

    Args:
        protein_df: Protein expression DataFrame
        missing_threshold: Remove proteins with >threshold fraction missing
        cv_threshold: Remove proteins with coefficient of variation < threshold

    Returns:
        (filtered_df, removed_proteins)
    """
    # Missing value filter
    missing_frac = protein_df.isnull().mean()
    pass_missing = missing_frac[missing_frac <= missing_threshold].index

    # Low variation filter (coefficient of variation)
    sub = protein_df[pass_missing]
    cv = sub.std() / (sub.mean().abs() + 1e-8)
    pass_cv = cv[cv >= cv_threshold].index.tolist()

    removed = [c for c in protein_df.columns if c not in pass_cv]
    logger.info(
        f"Protein quality filter: kept {len(pass_cv)}, removed {len(removed)} proteins"
    )
    return protein_df[pass_cv], removed


def stability_report(
    selected_features_per_fold: list[list[str]],
) -> dict:
    """
    Compute feature-selection stability across cross-validation folds.

    Args:
        selected_features_per_fold: List of selected feature lists per fold

    Returns:
        Dict with stability metrics and consistent features
    """
    from collections import Counter

    n_folds = len(selected_features_per_fold)
    all_features = [f for fold in selected_features_per_fold for f in fold]
    freq = Counter(all_features)

    # Features selected in all folds
    consistent = [f for f, cnt in freq.items() if cnt == n_folds]
    # Features selected in >50% folds
    majority = [f for f, cnt in freq.items() if cnt > n_folds / 2]

    # Jaccard stability index (average pairwise)
    jaccard_scores = []
    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            s1 = set(selected_features_per_fold[i])
            s2 = set(selected_features_per_fold[j])
            if s1 | s2:
                jaccard_scores.append(len(s1 & s2) / len(s1 | s2))
    avg_jaccard = float(np.mean(jaccard_scores)) if jaccard_scores else 0.0

    return {
        "n_folds": n_folds,
        "consistent_features": consistent,
        "majority_features": majority,
        "avg_jaccard_stability": round(avg_jaccard, 4),
        "feature_frequency": dict(freq.most_common(20)),
    }
