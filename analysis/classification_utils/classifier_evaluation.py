import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from analysis.classification_utils.metrics import (
    calculate_precision,
    calculate_recall,
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


class ProbaLogisticRegression(LogisticRegression):
    """
    Logistic regression classifier that predicts based on a probability threshold.

    Parameters
    ----------
    proba_threshold : float
        The probability threshold above which the classifier predicts a positive class.
    """

    def __init__(self, proba_threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.proba_threshold = proba_threshold

    def predict(self, X):
        probabilities = self.predict_proba(X)
        y_pred = (probabilities[:, 1] >= self.proba_threshold).astype(int)
        return y_pred


class ProbaXGBClassifier(XGBClassifier):
    """
    XGBoost classifier that predicts based on a probability threshold.

    Parameters
    ----------
    proba_threshold : float
        The probability threshold above which the classifier predicts a positive class.
    """

    def __init__(self, proba_threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.proba_threshold = proba_threshold

    def predict(self, X):
        probabilities = self.predict_proba(X)
        y_pred = (probabilities[:, 1] >= self.proba_threshold).astype(int)
        return y_pred


def train_and_predict(
    feature_df: pd.DataFrame,
    classifier,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, object, object]:
    X = feature_df[features]
    y = feature_df["gt_label"]
    num_fish = y.sum()
    cv = StratifiedKFold(n_splits=min(num_fish, 10))  # Ensure each split has at least one positive sample

    y_pred = np.zeros(y.shape)
    confusion_matrices = []

    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Scaling the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        y_pred[test_index] = predictions
        confusion = confusion_matrix(y_test, predictions)
        confusion_matrices.append(confusion)

    # Train the final model on the entire dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    classifier.fit(X_scaled, y)

    summed_confusion = np.sum(confusion_matrices, axis=0)
    return summed_confusion, y_pred, classifier, scaler


def compute_metrics(confusion_matrix: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = calculate_precision(tp, fp)
    recall = calculate_recall(tp, fn)

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_score": f_beta(1, precision, recall),
        "F_1_2_score": f_beta(0.5, precision, recall),
        "F2_score": f_beta(2, precision, recall),
        "confusion_matrix": confusion_matrix,
    }
    return metrics


def train_and_evaluate_model(
    feature_df: pd.DataFrame,
    classifier,
    features: list[str],
    metrics_to_show: list = ["F_1_2_score"],
) -> tuple[dict[str, float], np.ndarray, object, object]:
    summed_confusion, y_pred, trained_classifier, trained_scaler = train_and_predict(feature_df, classifier, features)
    metrics = compute_metrics(summed_confusion)
    print({k: metrics[k] for k in metrics_to_show}, features)
    return metrics, y_pred, trained_classifier, trained_scaler
