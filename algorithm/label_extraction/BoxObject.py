# Code written by Leiv Andresen, HTD-A, leiv.andresen@axpo.com

from math import cos, sin

import cv2 as cv
import numpy as np


class BoxObject:
    def __init__(self, identifier, contour, frame_number):
        self.ID = identifier
        self.persistent_id = None
        self.frames_observed = [frame_number]
        self.show = [False]
        self.contours = [contour]

        x, y, w, h = cv.boundingRect(contour)
        self.area = [w * h]
        self.midpoints = [(int(x + w / 2), int(y + h / 2))]

        self.classifications = ["Objekt"]

        self.velocity = []
        self.mean_v = None
        self.mean_v_last_10 = None
        self.river_rot = -2.7229  # TOD Autodetect

    def update_object(self, detection):
        self.frames_observed.append(detection.frames_observed[-1])
        self.show.append(False)
        self.contours.append(detection.contours[-1])
        self.midpoints.append(detection.midpoints[-1])
        self.area.append(detection.area[-1])

        self.calculate_speed()
        # self.classify_object()

    def draw_bounding_box(self, img, color):
        x, y, w, h = cv.boundingRect(self.contours[-1])
        cv.rectangle(img, (x, y), (x + w, y + h), color, 3)

        cv.putText(
            img,
            (self.classifications[-1] + " " + str(self.persistent_id) + " Area: " + str(self.area[-1])),
            (x, y - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
        )
        return img

    def draw_classifications_box(self, img, downsampled=100):
        x, y, w, h = cv.boundingRect(self.contours[-1])
        upsample_factor = int(100 / downsampled)
        x, y, w, h = (
            x * upsample_factor,
            y * upsample_factor,
            w * upsample_factor,
            h * upsample_factor,
        )
        if self.classifications[-1] == "Fisch":
            color = (0, 255, 0)
        else:
            color = (250, 150, 150)

        cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
        if downsampled != 100:
            cv.putText(
                img,
                (self.classifications[-1]),
                (x, y - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
            )

        return img

    def calculate_speed(self):
        frame_diff = float(self.frames_observed[-1] - self.frames_observed[-2])
        if frame_diff == 0:
            return
        v_x = float(self.midpoints[-1][0] - self.midpoints[-2][0]) / frame_diff
        v_y = float(self.midpoints[-1][1] - self.midpoints[-2][1]) / frame_diff
        v_rot = self.rotate_vector(np.array([v_x, v_y]), theta=-self.river_rot)
        self.velocity.append(v_rot)
        self.mean_v = np.median(np.asarray(self.velocity), axis=0)
        if len(self.velocity) > 11:
            self.mean_v_last_10 = np.median(np.asarray(self.velocity[-10:]), axis=0)

    def draw_past_midpoints(self, img, color):
        for point in self.midpoints:
            cv.circle(img, point, 2, color, -1)
        return img

    def occurences_in_last_x(self, frame_number, x):
        a = np.array(self.frames_observed, dtype="int")
        return a[a >= frame_number - x].shape[0]

    def rotate_vector(self, vec, theta):
        rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
        return np.dot(rot, vec)
