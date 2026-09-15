"""
SHAP-based explainability for pCR predictions.
"""
from __future__ import annotations
import logging
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """SHAP explainer for sklearn/xgboost models."""

    def __init__(self, model, feature_names: list[str], model_type: str = "tree"):
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.explainer = None

    def fit(self, X_background: pd.DataFrame) -> "SHAPExplainer":
        try:
            import shap
        except ImportError:
            raise ImportError("shap required: pip install shap")

        logger.info(f"Building SHAP explainer ({self.model_type}) on {len(X_background)} background samples...")
        if self.model_type == "tree":
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_type == "linear":
            self.explainer = shap.LinearExplainer(self.model, X_background)
        else:
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, shap.sample(X_background, 100)
            )
        return self

    def explain_global(self, X: pd.DataFrame, max_features: int = 30) -> list[dict]:
        """Compute global SHAP feature importance."""
        import shap
        shap_vals = self.explainer.shap_values(X)
        # For binary classifiers TreeExplainer returns list [class0, class1]
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        mean_abs = np.abs(shap_vals).mean(axis=0)
        mean_signed = shap_vals.mean(axis=0)

        features = []
        for i in np.argsort(mean_abs)[::-1][:max_features]:
            features.append({
                "name": self.feature_names[i],
                "mean_abs_shap": float(mean_abs[i]),
                "mean_shap": float(mean_signed[i]),
                "type": "positive" if mean_signed[i] >= 0 else "negative",
            })
        logger.info(f"Global SHAP: top feature = {features[0]['name']} ({features[0]['mean_abs_shap']:.4f})")
        return features

    def explain_patient(self, x: pd.Series) -> list[dict]:
        """Compute SHAP values for a single patient."""
        import shap
        x_arr = x.values.reshape(1, -1)
        shap_vals = self.explainer.shap_values(x_arr)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        sv = shap_vals[0]

        features = []
        for i in np.argsort(np.abs(sv))[::-1]:
            features.append({
                "name": self.feature_names[i],
                "shap_value": float(sv[i]),
                "feature_value": float(x_arr[0, i]),
                "type": "positive" if sv[i] >= 0 else "negative",
            })
        return features

    def get_top_biomarkers(
        self,
        X: pd.DataFrame,
        n_positive: int = 10,
        n_negative: int = 10,
    ) -> dict:
        """Get top positive and negative biomarkers by mean SHAP value."""
        global_features = self.explain_global(X, max_features=len(self.feature_names))
        positives = [f for f in global_features if f["mean_shap"] > 0][:n_positive]
        negatives = sorted(
            [f for f in global_features if f["mean_shap"] < 0],
            key=lambda x: x["mean_shap"]
        )[:n_negative]
        return {
            "top_positive": positives,
            "top_negative": negatives,
            "all_features": global_features,
        }
