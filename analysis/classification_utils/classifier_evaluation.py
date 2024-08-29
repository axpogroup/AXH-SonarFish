import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from analysis.classification_utils.metrics import (
    calculate_f1_score,
    calculate_precision,
    f_beta,
)


class RandomClassifier:

    def __init__(self, class_counts: dict[int, int]):
        total_count = sum(class_counts.values())
        self.class_probabilities = {k: v / total_count for k, v in class_counts.items()}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        # draw random samples from distribuiton of classes
        classes = list(self.class_probabilities.keys())
        probabilities = list(self.class_probabilities.values())
        return np.random.choice(classes, size=X.shape[0], p=probabilities)


def train_and_predict(feature_df: pd.DataFrame, classifier, features: list[str]) -> np.ndarray:
    X = feature_df[features]
    y = feature_df["gt_label"]
    num_fish = y.sum()
    cv = StratifiedKFold(n_splits=min(num_fish, 10))  # Ensure each split has at least one positive sample

    y_pred = np.zeros(y.shape)
    confusion_matrices = []

    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        y_pred[test_index] = predictions
        confusion = confusion_matrix(y_test, predictions)
        confusion_matrices.append(confusion)

    summed_confusion = np.sum(confusion_matrices, axis=0)
    return summed_confusion, y_pred


def compute_metrics(confusion_matrix: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = calculate_precision(tp, fp)
    recall = tp / (tp + fn)
    f1_score = calculate_f1_score(precision, recall)
    f_1_2_score = f_beta(0.5, precision, recall)

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_score": f1_score,
        "F_1_2_score": f_1_2_score,
        "confusion_matrix": confusion_matrix,
    }
    return metrics


def train_and_evaluate_model(
    feature_df: pd.DataFrame,
    classifier,
    features: list[str],
    metrics_to_show: list = ["F_1_2_score"],
) -> dict[str, float]:
    summed_confusion, y_pred = train_and_predict(feature_df, classifier, features)
    metrics = compute_metrics(summed_confusion)
    print({k: metrics[k] for k in metrics_to_show}, features)
    return metrics, y_pred
