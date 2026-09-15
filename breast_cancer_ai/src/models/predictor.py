"""
End-to-end pCR predictor — wraps the full inference pipeline.
Takes patient data, returns prediction + probability.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)


class pCRPredictor:
    """
    End-to-end pCR predictor.
    Loads a trained model artifact and runs inference on new patient data.
    """

    def __init__(self, model_dir: str = "experiments/latest"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.feature_names: list[str] = []
        self.model_type: str = "unknown"
        self.threshold: float = 0.5

    def load(self) -> "pCRPredictor":
        """Load trained model artifacts from disk."""
        meta_path = self.model_dir / "predictor_meta.json"
        model_path = self.model_dir / "predictor_model.joblib"

        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {model_path}. "
                "Run the full pipeline first."
            )

        import json
        with open(meta_path) as f:
            meta = json.load(f)

        self.model = joblib.load(model_path)
        self.feature_names = meta.get("feature_names", [])
        self.model_type = meta.get("model_type", "unknown")
        self.threshold = meta.get("threshold", 0.5)

        scaler_path = self.model_dir / "predictor_scaler.joblib"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)

        logger.info(f"Loaded predictor: {self.model_type} from {self.model_dir}")
        return self

    def predict_patient(self, patient_features: dict) -> dict:
        """
        Run inference for a single patient.

        Args:
            patient_features: Dict mapping feature_name → value

        Returns:
            dict with 'prediction' (0/1), 'probability' (float), 'prediction_label' (str)
        """
        if self.model is None:
            raise RuntimeError("Predictor not loaded. Call .load() first.")

        # Build feature vector in correct order
        x = np.array([
            float(patient_features.get(f, 0.0))
            for f in self.feature_names
        ]).reshape(1, -1)

        if self.scaler is not None:
            x = self.scaler.transform(x)

        prob = float(self.model.predict_proba(x)[0, 1])
        pred = int(prob >= self.threshold)

        return {
            "prediction": pred,
            "probability": round(prob, 4),
            "prediction_label": "pCR" if pred == 1 else "No pCR",
            "model_type": self.model_type,
            "threshold": self.threshold,
        }

    def predict_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        """Run inference on a batch of patients."""
        if self.model is None:
            raise RuntimeError("Predictor not loaded. Call .load() first.")

        available = [f for f in self.feature_names if f in X.columns]
        X_aligned = X.reindex(columns=self.feature_names, fill_value=0.0)

        X_np = X_aligned.values
        if self.scaler is not None:
            X_np = self.scaler.transform(X_np)

        probs = self.model.predict_proba(X_np)[:, 1]
        preds = (probs >= self.threshold).astype(int)

        return pd.DataFrame({
            "patient_id": X.index,
            "probability": probs.round(4),
            "prediction": preds,
            "prediction_label": ["pCR" if p == 1 else "No pCR" for p in preds],
        })

    def save(
        self,
        model,
        feature_names: list[str],
        model_type: str,
        scaler=None,
        threshold: float = 0.5,
        extra_meta: dict = None,
    ):
        """Save trained model artifacts."""
        import json
        self.model_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, self.model_dir / "predictor_model.joblib")

        meta = {
            "model_type": model_type,
            "feature_names": feature_names,
            "threshold": threshold,
            **(extra_meta or {}),
        }
        with open(self.model_dir / "predictor_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        if scaler is not None:
            joblib.dump(scaler, self.model_dir / "predictor_scaler.joblib")

        logger.info(f"Predictor saved to {self.model_dir}")
