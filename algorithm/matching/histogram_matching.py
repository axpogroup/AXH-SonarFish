import cv2 as cv
import numpy as np


def _histogram_cost(
    target_dets: list[np.array],
    feature_dets: list[np.array],
    target_dets_idx: list[int] = None,
    feature_dets_idx: list[int] = None,
) -> np.ndarray:
    """
    Calculates a cost matrix based on the area ratio between target detections and feature detections.

    Args:
        target_dets (list[Detection]): A list of image histogram of target detection objects.
        feature_dets (list[Detection]): A list of image histogram of feature detection objects.
        target_dets_idx (list[int], optional): A list of indices for the target detections. Defaults to None.
        feature_dets_idx (list[int], optional): A list of indices for the feature detections. Defaults to None.

    Returns:
        ndarray: A cost matrix of shape (len(track_indices), len(detection_indices)).
        Each entry (i, j) is defined as max(A1 / A2, A2 / A1) where A1 and A2 are the areas of the
        bounding boxes of the i-th track and j-th detection.
    """
    target_dets_idx = target_dets_idx if target_dets_idx is not None else range(len(target_dets))
    feature_dets_idx = feature_dets_idx if feature_dets_idx is not None else range(len(feature_dets))

    hist_similarity = np.zeros((len(target_dets), len(feature_dets)))
    for i, target in enumerate(target_dets):
        for j, feature in enumerate(feature_dets):
            hist_similarity[i, j] = cv.compareHist(target, feature, cv.HISTCMP_BHATTACHARYYA)

    return np.maximum(0.0, hist_similarity.min(axis=0))
