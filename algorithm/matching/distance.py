from functools import wraps

import numpy as np
from deepsort.nn_matching import _nn_cosine_distance, _nn_euclidean_distance

from algorithm.matching.area_matching import _area_cost
from algorithm.matching.histogram_matching import _histogram_cost


def feature_extractor(func, feature_to_extract: str):
    """
    Decorator to extract the required feature for a distance metric from
    samples and features, i.e., detections and tracks.

    Args:
        func (Callable): The original function that calculates the distance metric.
        feature_to_extract (str): The name of the feature to extract from the samples and features.

    Returns:
        Callable: The decorated function that extracts the specified feature before calculating the distance.
    """

    @wraps(func)
    def wrapper(samples, features):
        samples = [s[feature_to_extract] for s in samples]
        features = [f[feature_to_extract] for f in features]

        return func(samples, features)

    return wrapper


class DistanceMetric(object):
    metric_options = {
        "euclidean": feature_extractor(_nn_euclidean_distance, "center_pos"),
        "cosine": feature_extractor(_nn_cosine_distance, "center_pos"),
        "blob_area": feature_extractor(_area_cost, "contour"),
        "histogram": feature_extractor(_histogram_cost, "histogram"),
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
            raise ValueError(f"Invalid metric; must be one of {self.metric_options.keys()}")
        self.feature_keys = ["center_pos"]
        if metric == "blob_area":
            self.feature_keys = ["contour"]

        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(
        self,
        features: list[dict],
        targets: list[int],
        active_targets: list[int],
    ) -> None:
        """
        Update the distance metric with new data.

        Args:
            features: A list of DetectedObject feature dictionaries.
            targets: An integer array of associated target identities.
            active_targets: A list of targets that are currently present in the scene.
        """
        for feature_dict, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature_dict)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget :]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features: list[dict], targets: list[int]) -> np.ndarray:
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
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix
