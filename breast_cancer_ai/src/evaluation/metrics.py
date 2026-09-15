"""
Evaluation metrics — accuracy, F1, ROC-AUC, precision, recall, confusion matrix.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
    roc_curve, average_precision_score,
)
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "model",
) -> dict:
    """Compute full evaluation metrics for a binary classifier."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, int(cm[0, 0]))

    # ROC curve points
    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_prob)
    roc_points = [
        {"fpr": float(f), "tpr": float(t)}
        for f, t in zip(fpr_arr, tpr_arr)
    ]

    metrics = {
        "model": model_name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "avg_precision": round(float(average_precision_score(y_true, y_prob)), 4),
        "confusion_matrix": {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)},
        "roc_curve": roc_points,
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int(len(y_true) - y_true.sum()),
    }
    logger.info(
        f"[{model_name}] AUC={metrics['roc_auc']:.4f} | "
        f"F1={metrics['f1']:.4f} | Acc={metrics['accuracy']:.4f} | "
        f"Prec={metrics['precision']:.4f} | Rec={metrics['recall']:.4f}"
    )
    return metrics


def cross_validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
    n_seeds: int = 3,
    model_name: str = "model",
) -> dict:
    """
    Run stratified k-fold cross-validation with multiple seeds.
    Returns mean ± std for each metric.
    """
    from copy import deepcopy

    all_metrics = []
    for seed in range(n_seeds):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            m = deepcopy(model)
            m.fit(X_tr, y_tr)
            y_pred = m.predict(X_val)
            y_prob = m.predict_proba(X_val)[:, 1] if hasattr(m, 'predict_proba') else y_pred.astype(float)

            fold_metrics = {
                "seed": seed, "fold": fold,
                "accuracy": float(accuracy_score(y_val, y_pred)),
                "f1": float(f1_score(y_val, y_pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_val, y_prob)),
                "precision": float(precision_score(y_val, y_pred, zero_division=0)),
                "recall": float(recall_score(y_val, y_pred, zero_division=0)),
            }
            all_metrics.append(fold_metrics)

    df = pd.DataFrame(all_metrics)
    summary = {"model": model_name, "n_folds": n_folds, "n_seeds": n_seeds}
    for col in ["accuracy", "f1", "roc_auc", "precision", "recall"]:
        summary[f"{col}_mean"] = round(float(df[col].mean()), 4)
        summary[f"{col}_std"] = round(float(df[col].std()), 4)

    logger.info(
        f"CV [{model_name}]: AUC={summary['roc_auc_mean']:.4f}±{summary['roc_auc_std']:.4f} | "
        f"F1={summary['f1_mean']:.4f}±{summary['f1_std']:.4f}"
    )
    return summary


def optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find threshold that maximizes F1 score on a validation set."""
    thresholds = np.linspace(0.1, 0.9, 81)
    best_f1, best_thresh = 0.0, 0.5
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f = f1_score(y_true, y_pred, zero_division=0)
        if f > best_f1:
            best_f1, best_thresh = f, t
    return round(float(best_thresh), 3)
