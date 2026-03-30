"""
Evaluation metrics for behavioral fingerprinting / biometric authentication.

Primary metric: Equal Error Rate (EER)
    The EER is the operating point where FAR == FRR.
    Lower EER → better authentication system.

Supporting metrics: AUC-ROC, FAR, FRR, accuracy, F1.
"""

import numpy as np
from sklearn.metrics import (
    roc_curve,
    auc,
    accuracy_score,
    precision_recall_fscore_support,
    average_precision_score,
)
from typing import Dict, Optional, Tuple


def compute_eer(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) and its operating threshold.

    Returns
    -------
    eer       : float  – EER in [0, 1]; lower is better.
    threshold : float  – Decision boundary at the EER point.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr  # False Negative Rate == False Rejection Rate

    # Find the crossing point |FAR - FRR|
    diffs = np.abs(fpr - fnr)
    idx = int(np.argmin(diffs))

    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thresholds[idx])
    return eer, threshold


def compute_far_frr(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
) -> Tuple[float, float]:
    """
    False Accept Rate and False Rejection Rate at a given threshold.

    FAR = impostors incorrectly accepted / total impostors
    FRR = genuine users incorrectly rejected / total genuine
    """
    y_pred = (y_scores >= threshold).astype(int)

    genuine_mask = y_true == 1
    impostor_mask = y_true == 0

    frr = float((y_pred[genuine_mask] == 0).mean()) if genuine_mask.any() else 0.0
    far = float((y_pred[impostor_mask] == 1).mean()) if impostor_mask.any() else 0.0
    return far, frr


def evaluate_model(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute a comprehensive suite of evaluation metrics.

    Parameters
    ----------
    y_true   : Ground-truth labels (1 = genuine, 0 = impostor).
    y_scores : Continuous prediction scores (higher → more genuine).
    y_pred   : Optional hard predictions; derived from EER threshold if omitted.

    Returns
    -------
    dict with keys: eer, eer_threshold, far_at_eer, frr_at_eer,
                    auc_roc, avg_precision, accuracy, precision,
                    recall, f1, n_genuine, n_impostor.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)

    eer, eer_thresh = compute_eer(y_true, y_scores)
    far, frr = compute_far_frr(y_true, y_scores, eer_thresh)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = float(auc(fpr, tpr))
    avg_prec = float(average_precision_score(y_true, y_scores))

    if y_pred is None:
        y_pred = (y_scores >= eer_thresh).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    return {
        "eer": eer,
        "eer_threshold": eer_thresh,
        "far_at_eer": far,
        "frr_at_eer": frr,
        "auc_roc": roc_auc,
        "avg_precision": avg_prec,
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "n_genuine": int((y_true == 1).sum()),
        "n_impostor": int((y_true == 0).sum()),
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    """One-line summary string for quick console output."""
    return (
        f"EER={metrics.get('eer', 0):.4f}  "
        f"AUC={metrics.get('auc_roc', 0):.4f}  "
        f"F1={metrics.get('f1', 0):.4f}  "
        f"Acc={metrics.get('accuracy', 0):.4f}  "
        f"FAR={metrics.get('far_at_eer', 0):.4f}  "
        f"FRR={metrics.get('frr_at_eer', 0):.4f}"
    )
