import cv2
import numpy as np


def calculate_fft_similarity(target, feature):
    # Shift the zero-frequency component to the center of the spectrum
    fft_shift_target = np.fft.fftshift(target)
    fft_shift_feature = np.fft.fftshift(feature)

    # Calculate the magnitude spectrum of the FFT
    magnitude_spectrum_target = np.abs(fft_shift_target)
    magnitude_spectrum_feature = np.abs(fft_shift_feature)

    # Normalize the magnitude spectrums
    magnitude_spectrum_target = (magnitude_spectrum_target - np.min(magnitude_spectrum_target)) / (
        np.max(magnitude_spectrum_target) - np.min(magnitude_spectrum_target)
    )
    magnitude_spectrum_feature = (magnitude_spectrum_feature - np.min(magnitude_spectrum_feature)) / (
        np.max(magnitude_spectrum_feature) - np.min(magnitude_spectrum_feature)
    )

    # Flatten the magnitude spectrums
    magnitude_spectrum_target = magnitude_spectrum_target.flatten()
    magnitude_spectrum_feature = magnitude_spectrum_feature.flatten()

    # Calculate the similarity score
    metric_val = cv2.compareHist(
        np.float32(magnitude_spectrum_target), np.float32(magnitude_spectrum_feature), cv2.HISTCMP_BHATTACHARYYA
    )
    return round(metric_val, 2)


def _fft_cost(
    target_dets: list[np.ndarray],
    feature_dets: list[np.ndarray],
    target_dets_idx: list[int] = None,
    feature_dets_idx: list[int] = None,
) -> np.ndarray:
    """
    Calculates a cost matrix based on the FFT similarity between target detections and feature detections.

    Args:
        target_dets (list[np.array]): A list of 2D arrays of the ffts of image patches.
        feature_dets (list[np.array]): A list of 2D arrays of the ffts of image patches.
        target_dets_idx (list[int], optional): A list of indices for the target detections. Defaults to None.
        feature_dets_idx (list[int], optional): A list of indices for the feature detections. Defaults to None.

    Returns:
        ndarray: A cost matrix of shape (len(target_dets), len(feature_dets)).
        Each entry (i, j) is the FFT similarity between the i-th target and j-th feature.
    """
    target_dets_idx = target_dets_idx if target_dets_idx is not None else range(len(target_dets))
    feature_dets_idx = feature_dets_idx if feature_dets_idx is not None else range(len(feature_dets))

    fft_similarity = np.zeros((len(target_dets), len(feature_dets)))
    for i, target in enumerate(target_dets):
        for j, feature in enumerate(feature_dets):
            fft_similarity[i, j] = calculate_fft_similarity(target, feature)

    cost_matrix = 2 - fft_similarity
    return np.maximum(0.0, cost_matrix.min(axis=0))
