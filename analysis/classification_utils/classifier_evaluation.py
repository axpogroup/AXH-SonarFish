from itertools import combinations
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
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
        self.class_counts = class_counts
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


def train_and_evaluate_model(
    feature_df: pd.DataFrame,
    classifier,
    features: list[str],
    metrics_to_show: Optional[list[str]] = None,
) -> dict[str, float]:
    X = feature_df[features]
    y = feature_df["gt_label"]
    num_fish = y.sum()
    cv = StratifiedKFold(n_splits=num_fish)

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
    tn, fp, fn, tp = summed_confusion.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = calculate_precision(tp, fp)
    recall = tp / (tp + fn)
    f1_score = calculate_f1_score(precision, recall)

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_score": f1_score,
        "F2_score": f_beta(2, precision, recall),
        "F3_score": f_beta(3, precision, recall),
        "F5_score": f_beta(5, precision, recall),
        "F8_score": f_beta(8, precision, recall),
        "F10_score": f_beta(10, precision, recall),
        "F_1_2_score": f_beta(0.5, precision, recall),
        "confusion_matris": summed_confusion,
    }
    if not metrics_to_show:
        metrics_to_show = metrics.keys()

    for k in metrics_to_show:
        print(f"{k}: {metrics[k]}")
    return metrics, y_pred, classifier


def find_best_feature_combination(
    feature_df: pd.DataFrame,
    classifier: object,
    all_features: list[str],
    performance_metric: str = "F_1_2_Score",
    max_features: Optional[int] = None,
    force_features: list[str] = [],
):
    """
    Takes a dataframe with features, a classifier, a list of all features to be explored, and a performance metric.
    Goes through all possible combinations of features and returns the best combination.

    :param feature_df: dataframe with features
    :param classifier: classifier object
    :param all_features: list of all features to be explored
    :param performance_metric: performance metric to be used
    :param max_features: maximum number of features to be used
    :param force_features: features to be included in all combinations

    :return: best feature combination and its score
    """
    all_features = list(set(all_features) - set(force_features))
    if max_features is None:
        max_features = len(all_features)

    best_score = -np.inf
    best_features = []

    for r in range(1, max_features + 1 - len(force_features)):
        for feature_combination in tqdm(combinations(all_features, r)):
            feature_combination = list(set(feature_combination) | set(force_features))
            metrics, _ = train_and_evaluate_model(
                feature_df, classifier, feature_combination, metrics_to_show={performance_metric: True}
            )
            score = metrics.get(performance_metric, -np.inf)

            if score > best_score:
                best_score = score
                best_features = feature_combination

            print(f"Current best score: {best_score} with features: {best_features}")

    return best_features, best_score


import pandas as pd


def greedy_feature_selection(
    feature_df: pd.DataFrame,
    classifier: object,
    all_features: list[str],
    performance_metric: str = "F_1_2_score",
    max_features: int = None,
    force_features: list[str] = [],
) -> tuple[list[str], float]:
    """
    Greedy algorithm to find the best feature combination by adding one feature at a time.

    :param feature_df: dataframe with features
    :param classifier: classifier object
    :param all_features: list of all features to be explored
    :param performance_metric: performance metric to be used
    :param max_features: maximum number of features to be used
    :param force_features: features to be included in all combinations

    :return: best feature combination and its score
    """

    if not max_features:
        max_features = len(all_features) - 1

    selected_features = force_features.copy()
    remaining_features = list(set(all_features) - set(force_features))
    best_score = -np.inf

    while len(selected_features) < max_features:
        best_feature = None
        for feature in remaining_features:
            current_features = selected_features + [feature]
            metrics, _, _ = train_and_evaluate_model(
                feature_df, classifier, current_features, metrics_to_show=[performance_metric]
            )
            score = metrics.get(performance_metric, -np.inf)

            if score > best_score:
                best_score = score
                best_feature = feature

        if best_feature is None:
            break

        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        print(f"Current best score: {best_score} with features: {selected_features}")

    return selected_features, best_score


def classifier_selection(
    feature_df: pd.DataFrame,
    classifiers: dict[str, object],
    features: list[str],
    target_metric: str = "F_1_2_score",
) -> tuple[object, list[str], float]:
    """
    Function to find the best classifier and feature combination.

    :param feature_df: dataframe with features
    :param classifiers: list of classifiers
    :param features: list of features
    :param target_metric: performance metric to be used

    :return: best classifier, best feature combination, and its score
    """
    best_score = -np.inf
    best_classifier = None

    for name, classifier in classifiers.items():
        print(f"Training {name}")
        metrics, _, trained_classifier = train_and_evaluate_model(
            feature_df,
            classifier,
            features,
        )
        if metrics[target_metric] > best_score:
            best_score = metrics[target_metric]
            best_classifier = trained_classifier
        print(f"{name} - {target_metric}: {metrics[target_metric]}")

    return best_classifier, best_score
