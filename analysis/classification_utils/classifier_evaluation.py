from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

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


class ProbaClassifier:
    """
    Takes a base classifier and a probability threshold to classify samples.

    Parameters:
    - base_estimator: The base classifier to use.
    - base_estimator_kwargs: Keyword arguments to pass to the base classifier.
    - proba_threshold: The probability threshold to use for classification.
        If fish probability is greater than this threshold, the sample is classified as fish.
    """

    def __init__(
        self,
        base_estimator: Union[
            LogisticRegression,
            RandomForestClassifier,
            SVC,
            GradientBoostingClassifier,
            AdaBoostClassifier,
            KNeighborsClassifier,
            XGBClassifier,
        ],
        base_estimator_kwargs: dict = {},
        proba_threshold: float = 0.5,
    ):
        self.base_estimator = base_estimator(**base_estimator_kwargs)
        self.proba_threshold = proba_threshold

    def fit(self, X, y):
        self.base_estimator.fit(X, y)

    def predict(self, X):
        probabilities = self.base_estimator.predict_proba(X)
        return (probabilities[:, 1] >= self.proba_threshold).astype(int)


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

        # Scaling the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

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

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_score": f1_score,
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
) -> dict[str, float]:
    summed_confusion, y_pred = train_and_predict(feature_df, classifier, features)
    metrics = compute_metrics(summed_confusion)
    print({k: metrics[k] for k in metrics_to_show}, features)
    return metrics, y_pred
