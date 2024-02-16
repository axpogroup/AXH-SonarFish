import numpy as np
from skimage.metrics import normalized_mutual_information


def _mutual_information_cost(
    target_dets: list[np.ndarray],
    feature_dets: list[np.ndarray],
    target_dets_idx: list[int] = None,
    feature_dets_idx: list[int] = None,
) -> np.ndarray:
    """
    Calculates a cost matrix based on the mutual information between target detections and feature detections.

    Args:
        target_dets (list[np.array]): A list of 2D arrays of single channel images for target detection objects.
        feature_dets (list[np.array]): A list of 2D arrays of single channel images for feature detection objects.
        target_dets_idx (list[int], optional): A list of indices for the target detections. Defaults to None.
        feature_dets_idx (list[int], optional): A list of indices for the feature detections. Defaults to None.

    Returns:
        ndarray: A cost matrix of shape (len(target_dets), len(feature_dets)).
        Each entry (i, j) is the mutual information between the i-th target and j-th feature.
    """
    target_dets_idx = target_dets_idx if target_dets_idx is not None else range(len(target_dets))
    feature_dets_idx = feature_dets_idx if feature_dets_idx is not None else range(len(feature_dets))

    mi_similarity = np.zeros((len(target_dets), len(feature_dets)))
    for i, target in enumerate(target_dets):
        for j, feature in enumerate(feature_dets):
            mi_similarity[i, j] = normalized_mutual_information(target, feature)

    cost_matrix = 2 - mi_similarity
    return np.maximum(0.0, cost_matrix.min(axis=0))
