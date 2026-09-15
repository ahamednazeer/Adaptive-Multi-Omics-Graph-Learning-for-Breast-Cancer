"""
LASSO-based feature selection for high-dimensional RNA/protein data.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, Lasso, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


class LASSOFeatureSelector:
    """
    LASSO-based feature selector with cross-validated alpha selection.
    Uses LogisticRegression with L1 penalty for classification tasks.
    """

    def __init__(
        self,
        cv_folds: int = 5,
        max_features: int = 500,
        random_state: int = 42,
        alphas: Optional[list[float]] = None,
    ):
        self.cv_folds = cv_folds
        self.max_features = max_features
        self.random_state = random_state
        self.alphas = alphas or [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        self.selected_features_: list[str] = []
        self.coef_: Optional[np.ndarray] = None
        self.best_C_: Optional[float] = None
        self.scaler_ = StandardScaler()

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "LASSOFeatureSelector":
        """
        Fit the LASSO selector on training data.

        Args:
            X: Feature matrix (patients x features)
            y: Binary target (pCR labels)

        Returns:
            self
        """
        logger.info(f"LASSO fitting on {X.shape[1]} features with {self.cv_folds}-fold CV...")

        # Scale features first
        X_scaled = self.scaler_.fit_transform(X)

        # Cross-validated logistic regression with L1 penalty
        # Cs = inverse of regularization strength (1/alpha)
        Cs = [1.0 / a for a in self.alphas]

        clf = LogisticRegressionCV(
            Cs=Cs,
            cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
            penalty="l1",
            solver="liblinear",
            max_iter=1000,
            random_state=self.random_state,
            class_weight="balanced",
            scoring="roc_auc",
        )
        clf.fit(X_scaled, y)

        self.best_C_ = float(clf.C_[0])
        self.coef_ = clf.coef_[0]

        # Select features with non-zero coefficients
        nonzero_mask = np.abs(self.coef_) > 1e-8
        selected_indices = np.where(nonzero_mask)[0]

        # If too many features, keep top-k by coefficient magnitude
        if len(selected_indices) > self.max_features:
            top_k = np.argsort(np.abs(self.coef_[selected_indices]))[::-1][: self.max_features]
            selected_indices = selected_indices[top_k]

        self.selected_features_ = [X.columns[i] for i in sorted(selected_indices)]

        logger.info(
            f"LASSO selected {len(self.selected_features_)} features "
            f"(best C={self.best_C_:.4f}, best_alpha={1/self.best_C_:.4f})"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select features from X using fitted selector."""
        available = [f for f in self.selected_features_ if f in X.columns]
        return X[available]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Return selected features sorted by coefficient magnitude."""
        if not self.selected_features_:
            raise RuntimeError("Selector not fitted yet.")
        feature_coefs = {
            f: float(self.coef_[i])
            for i, f in enumerate(
                [f for f in self.selected_features_]
            )
        }
        return pd.DataFrame(
            {
                "feature": list(feature_coefs.keys()),
                "coefficient": list(feature_coefs.values()),
                "abs_coefficient": [abs(v) for v in feature_coefs.values()],
            }
        ).sort_values("abs_coefficient", ascending=False)
