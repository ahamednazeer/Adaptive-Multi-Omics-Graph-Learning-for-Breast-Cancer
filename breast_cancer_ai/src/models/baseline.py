"""
Baseline models — Logistic Regression and XGBoost for pCR prediction.
Used to establish baseline performance before graph-based models.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
    roc_curve,
)

logger = logging.getLogger(__name__)


class LogisticRegressionBaseline:
    """Logistic Regression baseline with standard scaling."""

    def __init__(self, C: float = 1.0, random_state: int = 42):
        self.C = C
        self.random_state = random_state
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=C,
                max_iter=1000,
                random_state=random_state,
                class_weight="balanced",
                solver="lbfgs",
            )),
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticRegressionBaseline":
        logger.info(f"Training Logistic Regression on {X.shape} features...")
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)
        return _compute_metrics(y, y_pred, y_prob, model_name="LogisticRegression")


class XGBoostBaseline:
    """XGBoost baseline classifier."""

    def __init__(self, random_state: int = 42, **kwargs):
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost required: pip install xgboost")

        self.random_state = random_state
        params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "auc",
            "random_state": random_state,
            "tree_method": "hist",
            "n_jobs": -1,
        }
        params.update(kwargs)
        self.model = xgb.XGBClassifier(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostBaseline":
        import xgboost as xgb
        # Set class weight
        pos = y.sum()
        neg = len(y) - pos
        scale = neg / pos if pos > 0 else 1.0
        self.model.set_params(scale_pos_weight=scale)

        logger.info(f"Training XGBoost on {X.shape} features (scale_pos_weight={scale:.2f})...")
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)
        return _compute_metrics(y, y_pred, y_prob, model_name="XGBoost")

    def get_feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        imp = self.model.feature_importances_
        return pd.DataFrame(
            {"feature": feature_names, "importance": imp}
        ).sort_values("importance", ascending=False)


def _compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "model",
) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, cm[0, 0])

    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)
    roc_points = [{"fpr": round(float(f), 4), "tpr": round(float(t), 4)} for f, t in zip(fpr_arr, tpr_arr)]

    metrics = {
        "model": model_name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)},
        "roc_curve": roc_points,
    }
    logger.info(
        f"[{model_name}] AUC={metrics['roc_auc']:.4f} | "
        f"F1={metrics['f1']:.4f} | Acc={metrics['accuracy']:.4f}"
    )
    return metrics
