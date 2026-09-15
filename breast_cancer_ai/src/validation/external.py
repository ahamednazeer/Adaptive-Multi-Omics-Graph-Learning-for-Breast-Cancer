"""
External validation — uses GSE240671 pre-treatment samples for independent validation.
Pre/post analysis for treatment resistance biomarkers.
"""
from __future__ import annotations
import logging
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

logger = logging.getLogger(__name__)


def validate_on_external(
    gse240671_data: dict,
    model,
    feature_names: list[str],
    scaler=None,
) -> dict:
    """
    Validate trained model on GSE240671 pre-treatment samples.

    Args:
        gse240671_data: Output from load_gse240671()
        model: Trained sklearn-compatible model
        feature_names: Feature names the model expects
        scaler: Optional fitted scaler

    Returns:
        Validation metrics dict
    """
    pre = gse240671_data["pre"]
    expr = pre["expression"]
    meta = pre["meta"]

    if "pCR_Binary" not in meta.columns:
        logger.warning("No pCR_Binary label in GSE240671 pre-treatment data.")
        return {"error": "No target labels available"}

    y_true = meta["pCR_Binary"].values

    # Align features — use available intersection
    available = [f for f in feature_names if f in expr.columns]
    missing = [f for f in feature_names if f not in expr.columns]
    logger.info(
        f"External validation: {len(available)}/{len(feature_names)} features available "
        f"({len(missing)} missing → zero-filled)"
    )

    X = expr.reindex(columns=feature_names, fill_value=0.0).values

    if scaler is not None:
        X = scaler.transform(X)

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    metrics = {
        "dataset": "GSE240671",
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int(len(y_true) - y_true.sum()),
        "n_features_used": len(available),
        "n_features_missing": len(missing),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "roc_curve": [{"fpr": float(f), "tpr": float(t)} for f, t in zip(fpr, tpr)],
        "molecular_subtypes": meta["Molecular_Category"].value_counts().to_dict()
        if "Molecular_Category" in meta.columns else {},
    }
    logger.info(
        f"External validation: AUC={metrics['roc_auc']:.4f} | "
        f"F1={metrics['f1']:.4f} | Acc={metrics['accuracy']:.4f}"
    )
    return metrics


def analyze_resistance_biomarkers(
    gse240671_data: dict,
    biomarker_genes: list[str],
    top_n: int = 20,
) -> dict:
    """
    Compare pre vs post-treatment expression for selected biomarker genes.
    Identifies resistance-associated expression changes.

    Returns:
        Dict with per-gene pre/post stats and resistance direction
    """
    paired = gse240671_data["paired"]
    pre_expr = gse240671_data["pre"]["expression"]
    post_expr = gse240671_data["post"]["expression"]

    available = [g for g in biomarker_genes if g in pre_expr.columns and g in post_expr.columns]
    missing = [g for g in biomarker_genes if g not in available]
    logger.info(
        f"Resistance analysis: {len(available)}/{len(biomarker_genes)} genes available"
    )

    results = []
    for gene in available:
        pre_vals = pre_expr[gene].dropna()
        post_vals = post_expr[gene].dropna()

        pre_mean = float(pre_vals.mean())
        post_mean = float(post_vals.mean())
        fold_change = post_mean - pre_mean

        # Separate by pCR outcome
        pre_meta = gse240671_data["pre"]["meta"].reset_index(drop=True)
        pcr_mask = pre_meta["pCR_Binary"] == 1 if "pCR_Binary" in pre_meta.columns else pd.Series([True] * len(pre_meta))
        npcr_mask = ~pcr_mask

        pre_pcr_mean = float(pre_vals[pcr_mask.values[:len(pre_vals)]].mean()) if pcr_mask.sum() > 0 else 0.0
        pre_npcr_mean = float(pre_vals[npcr_mask.values[:len(pre_vals)]].mean()) if npcr_mask.sum() > 0 else 0.0

        results.append({
            "gene": gene,
            "pre_mean": round(pre_mean, 4),
            "post_mean": round(post_mean, 4),
            "fold_change": round(fold_change, 4),
            "direction": "up" if fold_change > 0 else "down",
            "pre_pcr_mean": round(pre_pcr_mean, 4),
            "pre_npcr_mean": round(pre_npcr_mean, 4),
            "pcr_vs_npcr_diff": round(pre_pcr_mean - pre_npcr_mean, 4),
            "resistance_associated": fold_change > 0 and pre_npcr_mean > pre_pcr_mean,
        })

    # Sort by absolute fold change
    results.sort(key=lambda x: abs(x["fold_change"]), reverse=True)

    return {
        "genes": results[:top_n],
        "total_analyzed": len(available),
        "unmapped": missing,
        "resistance_genes": [r for r in results if r["resistance_associated"]][:10],
    }
