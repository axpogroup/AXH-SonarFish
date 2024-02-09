from typing import Union

import numpy as np
from deepsort.nn_matching import _nn_cosine_distance, _nn_euclidean_distance

from algorithm.matching.area_matching import _area_cost


class DistanceMetric(object):
    metric_options = {
        "euclidean": _nn_euclidean_distance,
        "cosine": _nn_cosine_distance,
        "blob_area": _area_cost,
    }

    def __init__(self, metric: str, matching_threshold: float, budget: int = None):
        """
        Initialize the DistanceMetric object.

        Args:
            metric: The distance metric to use. One of mectric_options.
            matching_threshold: The matching threshold. Samples with larger distance are considered an
            invalid match.
            budget: If not None, fix samples per class to at most this number. Removes
            the oldest samples when the budget is reached.
        """
        try:
            self._metric = self.metric_options[metric]
        except KeyError:
            raise ValueError(
                f"Invalid metric; must be one of {self.metric_options.keys()}"
            )
        self.feature_keys = ["center_pos"]
        if metric == "blob_area":
            self.feature_keys = ["contour"]

        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def extract_features(
        self, features: Union[list[dict[str, np.ndarray]], dict[str, np.ndarray]]
    ) -> Union[list[np.ndarray], np.ndarray]:
        """
        Extracts the relevant features from the input.

        Args:
            features: The input features.

        Returns:
            The extracted features.
        """
        if isinstance(features, dict):
            features = [features]
        features = [feature[self.feature_keys[0]] for feature in features]
        if self.feature_keys[0] == "center_pos":
            return np.array(features)
        else:
            return features

    def partial_fit(
        self,
        features: Union[list[np.ndarray], np.ndarray],
        targets: list[int],
        active_targets: list[int],
    ) -> None:
        """
        Update the distance metric with new data.

        Args:
            features: An NxM matrix of N features of dimensionality M.
            targets: An integer array of associated target identities.
            active_targets: A list of targets that are currently present in the scene.
        """
        features = self.extract_features(features)
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget :]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(
        self, features: list[dict[str, np.ndarray]], targets: list[int]
    ) -> np.ndarray:
        """
        Compute distance between features and targets.

        Args:
            features: A list of N features of varying dimensionality.
            targets: A list of targets to match the given `features` against.

        Returns:
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.
        """
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(
                self.samples[target], self.extract_features(features)
            )
        return cost_matrix
