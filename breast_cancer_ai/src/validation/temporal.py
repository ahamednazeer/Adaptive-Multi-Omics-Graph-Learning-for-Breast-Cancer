"""
Temporal validation — tracks selected biomarker expression across T1→T4 timepoints.
Uses GSE122630 longitudinal dataset. Never used for training.
"""
from __future__ import annotations
import logging
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def analyze_temporal_trends(
    gse122630_data: dict,
    biomarker_genes: list[str],
    ensembl_to_symbol: Optional[dict] = None,
) -> dict:
    """
    Analyze expression trends of selected biomarkers across timepoints T1→T4.

    Args:
        gse122630_data: Output from load_gse122630()
        biomarker_genes: Gene symbols selected by LASSO/Boruta from ISPY2
        ensembl_to_symbol: Dict mapping ENSG_ID → gene_symbol (for GSE122630)

    Returns:
        Dict with per-gene temporal trends, stats, and plot-ready data
    """
    by_tp = gse122630_data["by_timepoint"]
    gene_cols = gse122630_data["gene_cols"]

    # Build reverse map: symbol → ensembl_id (for GSE122630 columns)
    symbol_to_ensembl = {}
    if ensembl_to_symbol:
        for ensg, sym in ensembl_to_symbol.items():
            symbol_to_ensembl[sym] = ensg

    results = []
    mapped_genes = []
    unmapped_genes = []

    for gene in biomarker_genes:
        # Find the corresponding column in GSE122630
        col = symbol_to_ensembl.get(gene) if symbol_to_ensembl else gene
        if col is None or col not in gene_cols:
            unmapped_genes.append(gene)
            continue
        mapped_genes.append(gene)

        trend = {"gene": gene, "ensembl_id": col, "timepoints": {}}
        timepoint_means = []

        for tp in ["T1", "T2", "T3", "T4"]:
            tp_data = by_tp.get(tp, {})
            expr = tp_data.get("expression")
            if expr is not None and col in expr.columns:
                vals = expr[col].dropna()
                mean_val = float(vals.mean())
                std_val = float(vals.std())
                trend["timepoints"][tp] = {
                    "mean": round(mean_val, 4),
                    "std": round(std_val, 4),
                    "n": int(len(vals)),
                    "description": by_tp[tp]["meta"]["Timepoint_Description"].iloc[0]
                    if len(by_tp[tp]["meta"]) > 0 else tp,
                }
                timepoint_means.append((tp, mean_val))

        if len(timepoint_means) >= 2:
            # Direction: is the gene going up or down T1→last timepoint?
            t1_val = timepoint_means[0][1]
            last_val = timepoint_means[-1][1]
            trend["direction"] = "up" if last_val > t1_val else "down"
            trend["fold_change"] = round(last_val - t1_val, 4)
        results.append(trend)

    logger.info(
        f"Temporal analysis: {len(mapped_genes)}/{len(biomarker_genes)} genes mapped. "
        f"{len(unmapped_genes)} unmapped."
    )

    return {
        "genes": results,
        "mapped_count": len(mapped_genes),
        "unmapped_count": len(unmapped_genes),
        "unmapped_genes": unmapped_genes,
        "timepoints_analyzed": ["T1", "T2", "T3", "T4"],
    }


def compute_treatment_response_signature(
    gse122630_data: dict,
    top_genes_ensembl: list[str],
) -> pd.DataFrame:
    """
    Compute mean expression changes (T1→T4) for top biomarker genes.
    Returns a DataFrame suitable for heatmap visualization.
    """
    by_tp = gse122630_data["by_timepoint"]
    rows = []

    for tp in ["T1", "T2", "T3", "T4"]:
        expr = by_tp.get(tp, {}).get("expression")
        if expr is None:
            continue
        available = [g for g in top_genes_ensembl if g in expr.columns]
        if not available:
            continue
        means = expr[available].mean()
        row = {"timepoint": tp}
        row.update(means.to_dict())
        rows.append(row)

    return pd.DataFrame(rows).set_index("timepoint") if rows else pd.DataFrame()
