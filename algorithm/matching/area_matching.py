from typing import Union

import cv2 as cv
import numpy as np
from deepsort.detection import Detection

def _area_cost(
        target_dets: Union[list[Detection], Detection], 
        feature_dets: Union[list[Detection], Detection], 
        target_dets_idx: list[int] = None, 
        feature_dets_idx: list[int] = None,
    ) -> np.ndarray:
    """

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `max(1 - area(tracks[track_indices[i]]) / area(detections[detection_indices[j]])),
             1 - area(detections[detection_indices[j]]) / area(tracks[track_indices[i]])))`.

    """
    if not isinstance(target_dets, list):
        target_dets = [target_dets]
    if not isinstance(feature_dets, list):
        feature_dets = [feature_dets]
    
    target_dets_idx = target_dets_idx if target_dets_idx is not None else range(len(target_dets))
    feature_dets_idx = feature_dets_idx if feature_dets_idx is not None else range(len(feature_dets))
    target_area = np.array([cv.contourArea(target_dets[tidx]) for tidx in target_dets_idx])
    feature_area = np.array([cv.contourArea(feature_dets[fidx]) for fidx in feature_dets_idx])

    target_area = target_area[:, np.newaxis]  # reshape for broadcasting
    feature_area = feature_area[np.newaxis, :]  # reshape for broadcasting

    area_ratio = np.minimum(target_area / feature_area, feature_area / target_area)
            
    distances = 1 - area_ratio
    return np.maximum(0.0, distances.min(axis=0))
    
    