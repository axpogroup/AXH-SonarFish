from typing import Optional

import cv2 as cv
import numpy as np


class BoundingBox:
    def __init__(
        self,
        identifier: int,
        contour: np.ndarray,
        frame_number: int,
        label: Optional[int] = None,
        precalculated_feature: Optional[str] = None,
    ):
        self.frames_observed = [frame_number]
        self.x, self.y, self.w, self.h = contour if contour.shape == (4,) else cv.boundingRect(contour)
        self.x = int(self.x)
        self.y = int(self.y)
        self.w = int(self.w)
        self.h = int(self.h)
        self.bounding_boxes = [(self.w, self.h)]
        self.top_lefts_y = [self.y]
        self.top_lefts_x = [self.x]
        self.midpoints = [(int(self.x + self.w / 2), int(self.y + self.h / 2))]
        self._contour = contour
        self.frame_number = frame_number
        self.ID = identifier
        self.label = label
        self.precalculated_feature = precalculated_feature

    def update_object(self, detection_box):
        self.frames_observed.append(detection_box.frames_observed[-1])
        self.midpoints.append(detection_box.midpoints[-1])
        # ideally, we remove the notion of history from this class
        self.top_lefts_x.append(detection_box.top_lefts_x[-1])
        self.top_lefts_y.append(detection_box.top_lefts_y[-1])
        self.bounding_boxes.append(detection_box.bounding_boxes[-1])


class DetectedBlob(BoundingBox):
    def __init__(
        self,
        identifier: int,
        contour: np.ndarray,
        frame_number: int,
        frame: dict[str, np.ndarray],
        input_settings: dict,
    ):
        super().__init__(
            identifier,
            contour,
            frame_number,
        )
        self.stddevs_of_pixels_intensity = []
        self.means_of_pixels_intensity = []
        self.areas = [self.w * self.h if contour.shape == (4,) else cv.contourArea(contour)]
        self.feature_patch = [self.get_feature_patch(frame, "difference_thresholded")]
        self.calculate_average_pixel_intensity(frame["difference"])
        self.feature = {
            "center_pos": self.center_pos,
            "contour": self.contour,
            "detection_id": self.ID,
            "area": self.area,
            # "patch": self.get_feature_patch(frame, "difference_thresholded"),
            # "sift": self.sift_features(self.get_feature_patch(frame, "difference_thresholded")),
            "histogram": self.histogram(frame=frame),
            "fft": self.fft(frame=frame),
            "bbox_size_to_stddev_ratio": self.bbox_size_to_stddev_ratio(
                input_settings.get("bbox_size_to_stddev_ratio_threshold")
            ),
        }
        if input_settings.get("store_raw_image_patch"):
            self.raw_image_patch = [self.get_feature_patch(frame, "raw")]

    def update_object(self, detection):
        super().update_object(detection)
        self.areas.append(detection.areas[-1])
        self.means_of_pixels_intensity.append(detection.means_of_pixels_intensity[-1])
        self.stddevs_of_pixels_intensity.append(detection.stddevs_of_pixels_intensity[-1])
        if hasattr(self, "feature_patch"):
            self.feature_patch.append(detection.feature_patch[-1])
        if hasattr(self, "raw_image_patch"):
            self.raw_image_patch.append(detection.raw_image_patch[-1])
        self.feature = detection.feature

    def calculate_average_pixel_intensity(self, reference_frame: np.ndarray):
        detection_box = reference_frame[self.y : self.y + self.h, self.x : self.x + self.w]
        if 0 not in detection_box.shape:
            mean, stddev = cv.meanStdDev(detection_box)
            self.means_of_pixels_intensity.append(mean[0][0])
            self.stddevs_of_pixels_intensity.append(stddev[0][0])

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.array([self.x, self.y, self.w, self.h], dtype=float)
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def get_feature_patch(self, frame, processing_step: str) -> np.ndarray:
        return frame[processing_step][self.y : self.y + self.h, self.x : self.x + self.w]

    @property
    def center_pos(self):
        return self.midpoints[-1]

    @property
    def contour(self):
        return self._contour

    @property
    def area(self):
        return self.areas[-1]

    def histogram(self, frame: dict[str, np.ndarray]):
        hist_raw = (
            np.histogram(self.get_feature_patch(frame, "difference_thresholded"), bins=range(257))[0]
            .reshape(-1, 1)
            .astype(np.float32)
        )
        return cv.normalize(hist_raw, hist_raw, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

    def sift_features(self, feature_patch: np.ndarray):
        return cv.SIFT_create().detectAndCompute(feature_patch, None)

    def fft(self, frame: dict[str, np.ndarray]):
        feature_patch = self.get_feature_patch(frame, "difference_thresholded")
        try:
            patch_resized = cv.resize(feature_patch, (64, 64))
            return np.fft.fft2(patch_resized)
        except Exception as e:
            print(str(e))
            print(feature_patch)
            return None

    @property
    def mean_pixel_intensity(self):
        return self.means_of_pixels_intensity[-1]

    @property
    def stddev_of_pixel_intensity(self):
        return self.stddevs_of_pixels_intensity[-1]

    @property
    def tlwh(self):
        return np.array([self.x, self.y, self.w, self.h])

    def bbox_size_to_stddev_ratio(self, bbox_size_to_stddev_ratio_threshold):
        if len(self.stddevs_of_pixels_intensity) == 0 or len(self.areas) == 0:
            return None
        if self.stddevs_of_pixels_intensity[-1] == 0:
            return bbox_size_to_stddev_ratio_threshold
        return self.areas[-1] / self.stddevs_of_pixels_intensity[-1]


class KalmanTrackedBlob(DetectedBlob):
    def __init__(
        self,
        identifier: int,
        frame_number: int,
        contour: np.ndarray,
        frame: dict[str, np.ndarray],
        input_settings: dict,
        ellipse_angle: Optional[float] = None,
        ellipse_axes_lengths: Optional[tuple[int, int]] = None,
        detection_is_tracked: bool = False,
    ):
        super().__init__(
            identifier=identifier,
            contour=contour,
            frame_number=frame_number,
            frame=frame,
            input_settings=input_settings,
        )
        self.detection_is_tracked = detection_is_tracked
        self.ellipse_angles = [ellipse_angle]
        self.ellipse_axes_lengths_pairs = [ellipse_axes_lengths]
        self.velocities = []
        self.calculate_speed()

    def update_object(self, detection):
        super().update_object(detection)
        self.detection_is_tracked = detection.detection_is_tracked
        self.ellipse_angles.append(detection.ellipse_angles[-1])
        self.ellipse_axes_lengths_pairs.append(detection.ellipse_axes_lengths_pairs[-1])
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
