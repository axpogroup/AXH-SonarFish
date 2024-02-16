from typing import Optional

import cv2 as cv
import numpy as np
from deepsort.detection import Detection


class DetectedObject(Detection):
    def __init__(
        self,
        identifier: int,
        contour: np.ndarray,
        frame_number: int,
        frame_dict_history: Optional[dict[int, dict[str, np.array]]] = None,
        confidence: float = 0.9,
        ellipse_angle: Optional[float] = None,
        ellipse_axes_lengths: Optional[tuple[int, int]] = None,
        track_is_confirmed: bool = True,
    ):
        self.stddevs_of_pixels_intensity = []
        self.means_of_pixels_intensity = []
        self.ID = identifier
        self.detection_is_confirmed = track_is_confirmed
        self.ellipse_angles = [ellipse_angle]
        self.ellipse_axes_lengths_pairs = [ellipse_axes_lengths]
        self.frame_dict_history = frame_dict_history
        self.frames_observed = [frame_number]
        self._contour = contour
        x, y, w, h = contour if contour.shape == (4,) else cv.boundingRect(contour)
        self.top_lefts_x = [x]
        self.top_lefts_y = [y]
        self.midpoints = [(int(x + w / 2), int(y + h / 2))]
        self.bounding_boxes = [(w, h)]
        self.areas = [w * h if contour.shape == (4,) else cv.contourArea(contour)]
        self.velocities = []
        self.tlwh = np.array([x, y, w, h], dtype=float)
        self.confidence = confidence
        self.calculate_speed()
        if frame_dict_history and "difference" in frame_dict_history.get(frame_number).keys():
            self.calculate_average_pixel_intensity(
                frame_dict_history.get(frame_number)["difference"],
            )
        self.update_object(self)

    def _get_feature_patch(self, processing_step: str):
        x, y, w, h = self.tlwh.astype(int)
        return self.frame_dict_history[self.frames_observed[-1]][processing_step][y : y + h, x : x + w]

    @property
    def feature(self):
        return {
            "center_pos": self.center_pos,
            "contour": self.contour,
            "area": self.area,
            "patch": self._get_feature_patch("difference_thresholded"),
            "sift": self.sift_features,
            "histogram": self.histogram,
            "fft": self.fft,
        }

    @property
    def center_pos(self):
        return self.midpoints[-1]

    @property
    def contour(self):
        return self._contour

    @property
    def area(self):
        return self.areas[-1]

    @property
    def histogram(self):
        img = self._get_feature_patch("difference_thresholded")
        hist_raw = np.histogram(img, bins=range(257))[0].reshape(-1, 1).astype(np.float32)

        return cv.normalize(hist_raw, hist_raw, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

    @property
    def sift_features(self):
        patch = self._get_feature_patch("difference_thresholded")
        return cv.SIFT_create().detectAndCompute(patch, None)

    @property
    def fft(self):
        patch = self._get_feature_patch("difference_thresholded")
        patch_resized = cv.resize(patch, (64, 64))
        return np.fft.fft2(patch_resized)

    @property
    def mean_pixel_intensity(self):
        return self.means_of_pixels_intensity[-1]

    @property
    def stddev_of_pixel_intensity(self):
        return self.stddevs_of_pixels_intensity[-1]

    @property
    def bbox_size_to_stddev_ratio(self):
        if len(self.stddevs_of_pixels_intensity) == 0 or len(self.areas) == 0:
            return None
        return self.areas[-1] / self.stddevs_of_pixels_intensity[-1]

    def update_object(self, detection: Detection):
        self.detection_is_confirmed = detection.detection_is_confirmed
        self.ellipse_angles.append(detection.ellipse_angles[-1])
        self.ellipse_axes_lengths_pairs.append(detection.ellipse_axes_lengths_pairs[-1])
        self.frames_observed.append(detection.frames_observed[-1])
        self.midpoints.append(detection.midpoints[-1])
        self.top_lefts_x.append(detection.top_lefts_x[-1])
        self.top_lefts_y.append(detection.top_lefts_y[-1])
        self.bounding_boxes.append(detection.bounding_boxes[-1])
        self.areas.append(detection.areas[-1])
        self.calculate_speed()
        if len(detection.velocities) > 0:
            self.velocities.append(detection.velocities[-1])
        if len(detection.stddevs_of_pixels_intensity) > 0:
            self.means_of_pixels_intensity.append(detection.means_of_pixels_intensity[-1])
            self.stddevs_of_pixels_intensity.append(detection.stddevs_of_pixels_intensity[-1])

    def calculate_speed(self):
        # For the speed to be sensible (e.g. non-zero) it must be taken over a longer period of time
        # Find a past observation that is at least ~2 seconds ago
        past_observation_id = -2
        if len(self.frames_observed) > 2 and self.frames_observed[past_observation_id]:
            while float(self.frames_observed[-1] - self.frames_observed[past_observation_id]) < 20:
                if -past_observation_id + 1 > len(self.frames_observed):
                    self.velocities.append(np.array([9, 9]))
                    return
                past_observation_id -= 1

                frame_diff = float(self.frames_observed[-1] - self.frames_observed[past_observation_id])
                if frame_diff > 0:
                    v_x = float(self.midpoints[-1][0] - self.midpoints[past_observation_id][0]) / frame_diff
                    v_y = float(self.midpoints[-1][1] - self.midpoints[past_observation_id][1]) / frame_diff
                    self.velocities.append(np.array([v_x, v_y]))

    def calculate_average_pixel_intensity(self, reference_frames: np.ndarray):
        x, y, w, h = self.tlwh.astype(int)
        detection_box = reference_frames[y : y + h, x : x + w]  # noqa 4
        if 0 in detection_box.shape:
            print("detection_box is empty")
            return
        mean, stddev = cv.meanStdDev(detection_box)
        self.means_of_pixels_intensity.append(mean[0])
        self.stddevs_of_pixels_intensity.append(stddev[0])
