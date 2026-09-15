"""
STRING database loader — parses protein interaction network files.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# STRING DB column names
STRING_COLS = [
    "protein1",
    "protein2",
    "neighborhood",
    "neighborhood_transferred",
    "fusion",
    "cooccurence",
    "homology",
    "coexpression",
    "coexpression_transferred",
    "experiments",
    "experiments_transferred",
    "database",
    "database_transferred",
    "textmining",
    "textmining_transferred",
    "combined_score",
]


def load_string_network(
    path: str,
    confidence_threshold: int = 400,
    species_id: str = "9606",  # 9606 = Homo sapiens
) -> pd.DataFrame:
    """
    Load a STRING protein interaction network TSV file.

    The file can be in one of two formats:
    1. Full STRING format (protein1 protein2 ... combined_score)
    2. Simplified format (protein1 protein2 combined_score)

    Args:
        path: Path to STRING TSV file
        confidence_threshold: Minimum combined_score (0-1000) to include an edge
        species_id: NCBI taxonomy ID for the species prefix

    Returns:
        DataFrame with columns: gene1, gene2, combined_score
    """
    p = Path(path)
    if not p.exists():
        logger.warning(f"STRING file not found: {path}. Returning empty DataFrame.")
        return pd.DataFrame(columns=["gene1", "gene2", "combined_score"])

    logger.info(f"Loading STRING network from: {path}")

    try:
        df = pd.read_csv(path, sep=" ", low_memory=False)
    except Exception:
        df = pd.read_csv(path, sep="\t", low_memory=False)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Handle protein name format (e.g., "9606.ENSP00000...")
    for col in ["protein1", "protein2"]:
        if col in df.columns:
            df[col] = df[col].str.replace(f"{species_id}.", "", regex=False)

    # Filter by confidence
    if "combined_score" in df.columns:
        df = df[df["combined_score"] >= confidence_threshold].copy()
    else:
        logger.warning("No 'combined_score' column found. Keeping all interactions.")

    # Rename to gene1 / gene2
    rename_map = {}
    if "protein1" in df.columns:
        rename_map["protein1"] = "gene1"
    if "protein2" in df.columns:
        rename_map["protein2"] = "gene2"
    df = df.rename(columns=rename_map)

    logger.info(
        f"STRING network loaded: {len(df)} interactions "
        f"(threshold={confidence_threshold})"
    )
    return df[["gene1", "gene2", "combined_score"]]


def map_genes_to_string(
    gene_list: list[str],
    string_df: pd.DataFrame,
) -> dict:
    """
    Map a list of gene names to nodes in the STRING network.

    Returns:
        dict with keys: 'mapped', 'unmapped', 'mapping_rate'
    """
    all_genes = set(string_df["gene1"].tolist() + string_df["gene2"].tolist())
    mapped = [g for g in gene_list if g in all_genes]
    unmapped = [g for g in gene_list if g not in all_genes]
    rate = len(mapped) / len(gene_list) if gene_list else 0.0

    logger.info(
        f"Gene mapping: {len(mapped)}/{len(gene_list)} genes mapped "
        f"to STRING ({rate:.1%})"
    )
    return {
        "mapped": mapped,
        "unmapped": unmapped,
        "mapping_rate": round(rate, 4),
    }


def filter_to_gene_set(
    string_df: pd.DataFrame,
    gene_set: list[str],
) -> pd.DataFrame:
    """
    Filter STRING network to only include edges where both genes are in gene_set.
    """
    gene_set_s = set(gene_set)
    mask = string_df["gene1"].isin(gene_set_s) & string_df["gene2"].isin(gene_set_s)
    filtered = string_df[mask].copy()
    logger.info(
        f"Filtered STRING to gene set: {len(filtered)} edges "
        f"among {len(gene_set)} genes"
    )
    return filtered
