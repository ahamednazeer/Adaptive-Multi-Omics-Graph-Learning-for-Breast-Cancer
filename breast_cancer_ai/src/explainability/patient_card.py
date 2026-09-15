"""
Patient explanation card — structured explainability output for a single patient.
"""
from __future__ import annotations
from typing import Optional
import pandas as pd


def build_patient_card(
    patient_id,
    prediction: dict,
    shap_features: list[dict],
    clinical_data: Optional[dict] = None,
    treatment_data: Optional[dict] = None,
    pathway_mapping: Optional[dict] = None,
    top_n: int = 10,
) -> dict:
    """
    Build a structured patient explanation card.

    Returns a JSON-serializable dict with all explanation components.
    """
    # Split SHAP features into positive/negative
    pos_features = sorted(
        [f for f in shap_features if f["shap_value"] > 0],
        key=lambda x: x["shap_value"], reverse=True
    )[:top_n]
    neg_features = sorted(
        [f for f in shap_features if f["shap_value"] < 0],
        key=lambda x: x["shap_value"]
    )[:top_n]

    # Identify whether features are RNA, protein, clinical, or treatment
    def _categorize(name: str) -> str:
        if any(name.upper() == n for n in ["HER2", "HR", "AGE", "GRADE", "NODE"]):
            return "clinical"
        if "." in name and name[0].isupper():  # e.g. AKT.S473
            return "protein"
        return "rna"

    top_genes = [f for f in pos_features + neg_features if _categorize(f["name"]) == "rna"]
    top_proteins = [f for f in pos_features + neg_features if _categorize(f["name"]) == "protein"]
    top_clinical = [f for f in pos_features + neg_features if _categorize(f["name"]) == "clinical"]

    # Map to pathways if available
    gene_names = [f["name"] for f in top_genes]
    pathways = []
    if pathway_mapping:
        for gene in gene_names:
            if gene in pathway_mapping:
                pathways.append({"gene": gene, "pathway": pathway_mapping[gene]})

    card = {
        "patient_id": patient_id,
        "prediction": prediction.get("prediction_label", "Unknown"),
        "probability": prediction.get("probability", 0.0),
        "model_type": prediction.get("model_type", "unknown"),
        "explanation": {
            "top_positive_factors": pos_features,
            "top_negative_factors": neg_features,
            "top_rna_features": top_genes[:5],
            "top_protein_features": top_proteins[:5],
            "top_clinical_features": top_clinical[:5],
            "pathway_associations": pathways,
        },
        "clinical_context": clinical_data or {},
        "treatment_context": treatment_data or {},
    }
    return card
