from math import atan, cos, sin

import cv2 as cv
import numpy as np


class DetectedObject:
    def __init__(self, identifier, contour, frame_number, settings_dict):
        self.ID = identifier
        self.frames_observed = [frame_number]
        self.show = [False]
        self.contours = [contour]

        x, y, w, h = cv.boundingRect(contour)
        self.midpoints = [(int(x + w / 2), int(y + h / 2))]

        self.classifications = ["Objekt"]
        self.velocity = []
        self.median_v = None
        self.median_v_last_10 = None

        # TOD: eventually auto detect
        self.river_pixel_velocity = np.array(settings_dict["river_pixel_velocity"]) / (
            100 / settings_dict["downsample"]
        )
        self.river_abs_velocity = np.linalg.norm(self.river_pixel_velocity)
        # Rotate coordinate system to align x with the direction of the river - 180 deg + (90 deg - atan(x/y))
        self.flow_direction_rot = np.pi * 1.5 - atan(
            self.river_pixel_velocity[0] / self.river_pixel_velocity[1]
        )

        self.min_occurences_for_fish = settings_dict["min_occurences_for_fish"]

    def update_object(self, detection):
        self.frames_observed.append(detection.frames_observed[-1])
        self.show.append(False)
        self.contours.append(detection.contours[-1])
        self.midpoints.append(detection.midpoints[-1])

        self.calculate_speed()
        self.classify_object()

    def calculate_speed_old(self):
        frame_diff = float(self.frames_observed[-1] - self.frames_observed[-2])
        if frame_diff == 0:
            return

        v_x = float(self.midpoints[-1][0] - self.midpoints[-2][0]) / frame_diff
        v_y = float(self.midpoints[-1][1] - self.midpoints[-2][1]) / frame_diff

        v_rot = self.rotate_vector(np.array([v_x, v_y]), theta=-self.flow_direction_rot)
        # self.velocity.append(np.array([v_x, v_y]))
        self.velocity.append(v_rot)
        self.median_v = np.mean(np.asarray(self.velocity), axis=0)
        if len(self.velocity) > 11:
            self.median_v_last_10 = np.mean(np.asarray(self.velocity[-10:]), axis=0)

    def calculate_speed(self):
        # For the speed to be sensible it must me taken over a longer period of time
        past_observation_id = -2
        while (
            float(self.frames_observed[-1] - self.frames_observed[past_observation_id])
            < 20
        ):
            if -past_observation_id + 1 > len(self.frames_observed):
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

        v_rot = self.rotate_vector(np.array([v_x, v_y]), theta=-self.flow_direction_rot)
        # self.velocity.append(np.array([v_x, v_y]))
        self.velocity.append(v_rot)
        self.median_v = np.mean(np.asarray(self.velocity), axis=0)
        # if len(self.velocity) > 11:
        #     self.median_v_last_10 = np.mean(np.asarray(self.velocity[-10:]), axis=0)

    def occurences_in_last_x(self, frame_number, x):
        a = np.array(self.frames_observed, dtype="int")
        return a[a >= frame_number - x].shape[0]

    def classify_object(self):
        fish = False
        if (len(self.frames_observed) < self.min_occurences_for_fish) or (
            len(self.velocity) < 1
        ):
            self.classifications.append("Objekt")
            return

        # Short term - if the path changes a lot it is a fish
        if abs(self.velocity[-1][1]) > 0.4 * self.river_abs_velocity:
            fish = True
        elif (
            abs(self.velocity[-1][0] - self.river_abs_velocity)
            > 0.4 * self.river_abs_velocity
        ):
            fish = True

        # # Long term - if the short term path is linear then the entire path must be above a certain irregularity
        # # to be classified as a fish
        # if self.classifications[-1] == "Fisch":
        #     if abs(self.median_v[1]) > 0.2*self.river_abs_velocity:
        #         fish = True
        #     elif abs(self.median_v[0] - self.river_abs_velocity) > 0.4*self.river_abs_velocity:
        #         fish = True

        if fish:
            self.classifications.append("Fisch")
        else:
            self.classifications.append("Objekt")
        return

    def rotate_vector(self, vec, theta):
        rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
        return np.dot(rot, vec)
