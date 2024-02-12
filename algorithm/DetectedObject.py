import cv2 as cv
import numpy as np
from deepsort.detection import Detection


class DetectedObject(Detection):
    def __init__(
        self,
        identifier: int,
        contour: np.ndarray,
        frame_number: int,
        confidence: float = 0.9,
    ):
        self.ID = identifier
        self.frames_observed = [frame_number]
        x, y, w, h = contour if contour.shape == (4,) else cv.boundingRect(contour)
        self.top_lefts_x = [x]
        self.top_lefts_y = [y]
        self.midpoints = [(int(x + w / 2), int(y + h / 2))]
        self.bounding_boxes = [(w, h)]
        self.areas = [w * h if contour.shape == (4,) else cv.contourArea(contour)]
        self.velocities = [np.array([np.NAN, np.NAN])]

        self.tlwh = np.array([x, y, w, h], dtype=float)
        self.confidence = confidence
        self.feature = {
            "center_pos": self.midpoints[-1],
            "contour": contour,
            "area": self.areas[-1],
        }

    def update_object(self, detection: Detection):
        self.frames_observed.append(detection.frames_observed[-1])
        self.midpoints.append(detection.midpoints[-1])
        self.top_lefts_x.append(detection.top_lefts_x[-1])
        self.top_lefts_y.append(detection.top_lefts_y[-1])
        self.bounding_boxes.append(detection.bounding_boxes[-1])
        self.areas.append(detection.areas[-1])
        self.calculate_speed()

    def calculate_speed(self):
        # For the speed to be sensible (e.g. non-zero) it must be taken over a longer period of time
        # Find a past observation that is at least ~2 seconds ago
        past_observation_id = -2
        while (
            float(self.frames_observed[-1] - self.frames_observed[past_observation_id])
            < 20
        ):
            if -past_observation_id + 1 > len(self.frames_observed):
                self.velocities.append(np.array([np.NAN, np.NAN]))
                return
            past_observation_id -= 1

        frame_diff = float(
            self.frames_observed[-1] - self.frames_observed[past_observation_id]
        )
        v_x = (
            float(self.midpoints[-1][0] - self.midpoints[past_observation_id][0])
            / frame_diff
        )
        v_y = (
            float(self.midpoints[-1][1] - self.midpoints[past_observation_id][1])
            / frame_diff
        )
        return v_x, v_y
