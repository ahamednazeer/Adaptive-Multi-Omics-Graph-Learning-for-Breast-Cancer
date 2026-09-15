"""
Boruta feature selection — all-relevant feature selection using Random Forest.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


class BorutaSelector:
    """
    Boruta feature selection algorithm.
    Creates shadow features (shuffled copies), trains RF on combined dataset,
    and iteratively rejects features that underperform their shadows.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_iter: int = 100,
        alpha: float = 0.05,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.alpha = alpha
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.selected_features_: list[str] = []
        self.tentative_features_: list[str] = []
        self.rejected_features_: list[str] = []
        self.importance_history_: dict[str, list[float]] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BorutaSelector":
        """
        Run Boruta algorithm.

        Args:
            X: Feature matrix
            y: Binary target

        Returns:
            self
        """
        from scipy.stats import binomtest

        logger.info(
            f"Boruta starting: {X.shape[1]} features, "
            f"max_iter={self.max_iter}, alpha={self.alpha}"
        )

        n_features = X.shape[1]
        feature_names = list(X.columns)

        # Track hits (times feature beats best shadow)
        hits = np.zeros(n_features)
        trials = np.zeros(n_features)
        decided = np.zeros(n_features, dtype=bool)  # True if accepted or rejected
        accepted = np.zeros(n_features, dtype=bool)

        rng = np.random.default_rng(self.random_state)

        for iteration in range(self.max_iter):
            # Create shadow features (shuffled copies)
            X_shadow = X.copy()
            for col in feature_names:
                X_shadow[col] = rng.permutation(X_shadow[col].values)
            X_shadow.columns = [f"shadow_{c}" for c in X_shadow.columns]

            X_combined = pd.concat([X, X_shadow], axis=1)

            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state + iteration,
                n_jobs=self.n_jobs,
                class_weight="balanced",
                max_features="sqrt",
            )
            rf.fit(X_combined, y)

            importances = rf.feature_importances_
            orig_imp = importances[:n_features]
            shadow_imp = importances[n_features:]
            shadow_max = shadow_imp.max()

            # Update hits for undecided features
            for i in range(n_features):
                if not decided[i]:
                    trials[i] += 1
                    if orig_imp[i] > shadow_max:
                        hits[i] += 1

            # Binomial test to decide accept/reject
            for i in range(n_features):
                if decided[i]:
                    continue
                n_trials = int(trials[i])
                if n_trials == 0:
                    continue
                n_hits = int(hits[i])

                # Test if hit rate significantly > 0.5 (accept)
                p_accept = binomtest(n_hits, n_trials, p=0.5, alternative="greater").pvalue
                # Test if hit rate significantly < 0.5 (reject)
                p_reject = binomtest(n_hits, n_trials, p=0.5, alternative="less").pvalue

                if p_accept <= self.alpha / n_features:
                    accepted[i] = True
                    decided[i] = True
                elif p_reject <= self.alpha / n_features:
                    decided[i] = True

            # Store importance history
            for i, name in enumerate(feature_names):
                self.importance_history_.setdefault(name, []).append(float(orig_imp[i]))

            n_decided = decided.sum()
            if iteration % 10 == 0:
                logger.debug(
                    f"  Boruta iter {iteration+1}/{self.max_iter}: "
                    f"{accepted.sum()} accepted, {n_decided - accepted.sum()} rejected, "
                    f"{n_features - n_decided} tentative"
                )

            # Early stop if all decided
            if decided.all():
                logger.info(f"Boruta converged at iteration {iteration+1}")
                break

        self.selected_features_ = [feature_names[i] for i in range(n_features) if accepted[i]]
        self.tentative_features_ = [
            feature_names[i] for i in range(n_features) if not decided[i]
        ]
        self.rejected_features_ = [
            feature_names[i]
            for i in range(n_features)
            if decided[i] and not accepted[i]
        ]

        logger.info(
            f"Boruta complete: {len(self.selected_features_)} accepted, "
            f"{len(self.tentative_features_)} tentative, "
            f"{len(self.rejected_features_)} rejected"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with only selected features."""
        available = [f for f in self.selected_features_ if f in X.columns]
        return X[available]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Return mean importance across iterations for selected features."""
        records = []
        for feat in self.selected_features_:
            hist = self.importance_history_.get(feat, [])
            records.append(
                {
                    "feature": feat,
                    "mean_importance": float(np.mean(hist)) if hist else 0.0,
                    "std_importance": float(np.std(hist)) if hist else 0.0,
                    "status": "accepted",
                }
            )
        return pd.DataFrame(records).sort_values("mean_importance", ascending=False)
