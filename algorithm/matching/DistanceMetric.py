from typing import Union

import numpy as np
from deepsort.nn_matching import _nn_euclidean_distance, _nn_cosine_distance

from matching.area_matching import _area_cost

class DistanceMetric(object):
    """
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.

    Parameters
    ----------
    metric : str
        Either "euclidean" or "cosine".
    matching_threshold: float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
    budget : Optional[int]
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.

    Attributes
    ----------
    samples : Dict[int -> List[ndarray]]
        A dictionary that maps from target identities to the list of samples
        that have been observed so far.

    """

    def __init__(self, metric, matching_threshold, budget=None):
        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
            self.feature_keys = ['center_pos']
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
            self.feature_keys = ['center_pos']
        elif metric == "blob_area":
            self._metric = _area_cost
            self.feature_keys = ['contour']
        else:
            raise ValueError("Invalid metric; must be either 'euclidean', 'cosine' or 'blob_area'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}
        
    def extract_features(
            self, 
            features: Union[list[np.ndarray], dict[str, np.ndarray]]
        ) -> Union[list[np.ndarray], np.ndarray]:
        if isinstance(features, dict):
            features = [features]
        features = [feature[self.feature_keys[0]] for feature in features]
        if self.feature_keys[0] == 'center_pos':
            return np.array(features)
        else:
            return features
        
    def partial_fit(self, features, targets, active_targets):
        """Update the distance metric with new data.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : ndarray
            An integer array of associated target identities.
        active_targets : List[int]
            A list of targets that are currently present in the scene.

        """
        features = self.extract_features(features)
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget :]
        self.samples = {k: self.samples[k] for k in active_targets}
        
    def distance(self, features: list[dict[str, np.ndarray]], targets: list[int]) -> np.ndarray:
        """Compute distance between features and targets.

        Parameters
        ----------
        features : List[ndarray]
            A list of N features of varying dimensionality.
        targets : List[int]
            A list of targets to match the given `features` against.

        Returns
        -------
        ndarray
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.

        """
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], self.extract_features(features))
        return cost_matrix