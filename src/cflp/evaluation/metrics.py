"""
Automated evaluation metrics and experiment harness:
Calculates ROC-AUC, Accuracy, Precision, Recall, F1, Log Loss, and Brier Score.
"""
from typing import Dict, List, Tuple, Any
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
    log_loss,
    brier_score_loss
)

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes standard academic classification and calibration metrics.
    Robust against single-class edge cases.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred_probs = np.asarray(y_pred_probs, dtype=np.float32)

    # Clip probabilities to avoid log(0)
    y_pred_probs = np.clip(y_pred_probs, 1e-7, 1.0 - 1e-7)
    y_pred_binary = (y_pred_probs >= threshold).astype(np.int32)

    unique_classes = np.unique(y_true)
    if len(unique_classes) > 1:
        auc = float(roc_auc_score(y_true, y_pred_probs))
    else:
        auc = 0.5 # Undefined when single class present

    acc = float(accuracy_score(y_true, y_pred_binary))
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_binary, average="binary", zero_division=0
    )
    loss = float(log_loss(y_true, y_pred_probs))
    brier = float(brier_score_loss(y_true, y_pred_probs))

    # Positive class balance
    positive_rate = float(np.mean(y_true))

    return {
        "sample_count": int(len(y_true)),
        "positive_class_ratio": round(positive_rate, 4),
        "roc_auc": round(auc, 4),
        "accuracy": round(acc, 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1_score": round(float(f1), 4),
        "log_loss": round(loss, 4),
        "brier_score": round(brier, 4)
    }
