from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm
from sklearn.metrics import confusion_matrix


def f_beta(beta: float, precision: float, recall: float) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        f_beta_score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
        f_beta_score = np.nan_to_num(f_beta_score)  # Convert NaNs to zero
    return f_beta_score


def calculate_precision(tp, fp):
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.divide(tp, (tp + fp))
        precision = np.nan_to_num(precision)
    return precision


def calculate_recall(tp, fn):
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.divide(tp, (tp + fn))
        recall = np.nan_to_num(recall)
    return recall


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classifier_name: Optional[str] = None,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        norm=LogNorm(),
        xticklabels=["Predicted Negative", "Predicted Fish"],
        yticklabels=["Actual Negative", "Actual Fish"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix for {classifier_name}")
    plt.show()
