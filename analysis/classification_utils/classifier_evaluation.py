from itertools import combinations
from typing import Optional

from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

from analysis.classification_utils.metrics import f_beta, calculate_precision, calculate_f1_score


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


def train_and_evaluate_model(
    feature_df: pd.DataFrame,
    classifier,
    features: list[str],
    metrics_to_show: list = ["F_1_2_score"],
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
    f_1_2_score = f_beta(0.5, precision, recall)

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_score": f1_score,
        "F_1_2_score": f_1_2_score,
        "confusion_matris": summed_confusion,
    }

    print({k: metrics[k] for k in metrics_to_show}, features)

    return metrics, y_pred


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

    if max_features is None:
        max_features = len(all_features)

    selected_features = force_features.copy()
    remaining_features = list(set(all_features) - set(force_features))
    best_score = -np.inf

    while len(selected_features) < max_features:
        best_feature = None
        for feature in remaining_features:
            current_features = selected_features + [feature]
            metrics, _ = train_and_evaluate_model(
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
