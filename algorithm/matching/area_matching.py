from typing import Union

import cv2 as cv
import numpy as np
from deepsort.detection import Detection

def _area_cost(
        target_dets: list[Detection], 
        feature_dets: list[Detection], 
        target_dets_idx: list[int] = None, 
        feature_dets_idx: list[int] = None,
    ) -> np.ndarray:
    """
    Calculates a cost matrix based on the area ratio between target detections and feature detections.

    Args:
        target_dets (list[Detection]): A list of target detection objects.
        feature_dets (list[Detection]): A list of feature detection objects.
        target_dets_idx (list[int], optional): A list of indices for the target detections. Defaults to None.
        feature_dets_idx (list[int], optional): A list of indices for the feature detections. Defaults to None.

    Returns:
        ndarray: A cost matrix of shape (len(track_indices), len(detection_indices)). 
        Each entry (i, j) is defined as max(A1 / A2, A2 / A1) where A1 and A2 are the areas of the
        bounding boxes of the i-th track and j-th detection.
    """
    target_dets_idx = target_dets_idx if target_dets_idx is not None else range(len(target_dets))
    feature_dets_idx = feature_dets_idx if feature_dets_idx is not None else range(len(feature_dets))
    target_area = np.array([cv.contourArea(target_dets[tidx]) for tidx in target_dets_idx])
    feature_area = np.array([cv.contourArea(feature_dets[fidx]) for fidx in feature_dets_idx])

    target_area = target_area[:, np.newaxis]  # reshape for broadcasting
    feature_area = feature_area[np.newaxis, :]  # reshape for broadcasting

    area_ratio = np.minimum(target_area / feature_area, feature_area / target_area)
            
    distances = 1 - area_ratio
    return np.maximum(0.0, distances.min(axis=0))
    
    