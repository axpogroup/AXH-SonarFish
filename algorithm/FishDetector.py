import os

import cv2 as cv
import numpy as np
import pandas as pd
from deepsort.nn_matching import NearestNeighborDistanceMetric

from algorithm.flow_conditions import rotate_velocity_vectors
from algorithm.DetectedObject import DetectedObject
from algorithm.utils import get_elapsed_ms, resize_img
from algorithm.tracking_filters import nearest_neighbor, kalman


class FishDetector:
    def __init__(self, settings_dict):
        self.object_filter = None
        self.conf = settings_dict
        self.frame_number = 0
        self.latest_obj_index = 0

        # Masks
        self.non_object_space_mask = cv.imread(
            os.path.join(self.conf["mask_directory"], "fish.png"),
            cv.IMREAD_GRAYSCALE,
        )
        self.sonar_controls_mask = cv.imread(
            os.path.join(self.conf["mask_directory"], "full.png"),
            cv.IMREAD_GRAYSCALE,
        )

        # Enhancement
        self.framebuffer = None
        self.mean_buffer = None
        self.mean_buffer_counter = None
        self.short_mean_float = None
        self.long_mean_float = None

    def detect_objects(self, raw_frame):
        start = cv.getTickCount()
        runtimes_ms = {}
        frame_dict = {}
        frame_dict["raw"] = raw_frame

        # Image enhancement
        if self.conf["downsample"]:
            frame_dict["raw_downsampled"] = resize_img(
                raw_frame, self.conf["downsample"]
            )
            frame_dict["gray"] = self.rgb_to_gray(frame_dict["raw_downsampled"])
        else:
            frame_dict["gray"] = self.rgb_to_gray(frame_dict["raw"])

        enhanced_temp = self.mask_regions(frame_dict["gray"], area="sonar_controls")
        enhanced_temp = cv.convertScaleAbs(
            enhanced_temp, alpha=self.conf["contrast"], beta=self.conf["brightness"]
        )
        self.update_buffers_calculate_means(enhanced_temp)
        frame_dict["gray_boosted"] = enhanced_temp

        if self.long_mean_float is None:
            runtimes_ms["enhance"] = get_elapsed_ms(start)
            runtimes_ms["detection_tracking"] = (
                get_elapsed_ms(start) - runtimes_ms["enhance"]
            )
            runtimes_ms["total"] = get_elapsed_ms(start)
            self.frame_number += 1
            return {}, frame_dict, runtimes_ms
        else:
            enhanced_temp = (self.short_mean_float - self.long_mean_float).astype(
                "int16"
            )
            frame_dict["long_mean"] = self.long_mean_float.astype("uint8")
            frame_dict["short_mean"] = self.short_mean_float.astype("uint8")
            frame_dict["difference"] = (enhanced_temp + 127).astype("uint8")
            frame_dict["absolute_difference"] = (abs(enhanced_temp) + 127).astype(
                "uint8"
            )

            adaptive_threshold = self.conf["difference_threshold_scaler"] * cv.blur(
                self.long_mean_float.astype("uint8"), (10, 10)
            )
            enhanced_temp[abs(enhanced_temp) < adaptive_threshold] = 0
            frame_dict["difference_thresholded"] = (enhanced_temp + 127).astype("uint8")

            enhanced_temp = (abs(enhanced_temp) + 127).astype("uint8")
            frame_dict["difference_thresholded_abs"] = enhanced_temp
            median_filter_kernel_px = self.mm_to_px(
                self.conf["median_filter_kernel_mm"]
            )
            enhanced_temp = cv.medianBlur(
                enhanced_temp, self.ceil_to_odd_int(median_filter_kernel_px)
            )
            frame_dict["median_filter"] = enhanced_temp
            runtimes_ms["enhance"] = get_elapsed_ms(start)

            # Threshold to binary
            ret, thres = cv.threshold(
                enhanced_temp, 127 + self.conf["difference_threshold_scaler"], 255, 0
            )
            ret, thres_raw = cv.threshold(
                frame_dict["difference_thresholded_abs"],
                127 + self.conf["difference_threshold_scaler"],
                255,
                0,
            )
            frame_dict["binary"] = thres
            # frame_dict["raw_binary"] = thres_raw
            dilation_kernel_px = self.mm_to_px(self.conf["dilation_kernel_mm"])
            kernel = cv.getStructuringElement(
                cv.MORPH_ELLIPSE,
                (
                    self.ceil_to_odd_int(dilation_kernel_px),
                    self.ceil_to_odd_int(dilation_kernel_px),
                ),
            )
            frame_dict["dilated"] = cv.dilate(thres, kernel, iterations=1)
            # frame_dict["closed"] = cv.morphologyEx(thres, cv.MORPH_CLOSE, kernel)
            # frame_dict["opened"] = cv.morphologyEx(thres, cv.MORPH_OPEN, kernel)
            # frame_dict["internal_external"] = (
            #     frame_dict["dilated"] - frame_dict["raw_binary"]
            # )

            # Extract keypoints
            contours, hier = cv.findContours(
                frame_dict["dilated"], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )
            detections = {}
            for contour in contours:
                self.latest_obj_index += 1
                new_object = DetectedObject(
                    self.latest_obj_index, contour, self.frame_number
                )
                detections[new_object.ID] = new_object

            runtimes_ms["detection_tracking"] = (
                get_elapsed_ms(start) - runtimes_ms["enhance"]
            )

        runtimes_ms["total"] = get_elapsed_ms(start)
        self.frame_number += 1
        return detections, frame_dict, runtimes_ms

    def associate_detections(self, detections, object_history):
        if len(detections) == 0:
            return object_history

        if self.conf['tracking_method'] == 'nearest_neighbor':
            max_association_distance_px = self.mm_to_px(
                self.conf["max_association_dist_mm"]
            )
            return nearest_neighbor.associate_detections(
                detections, 
                object_history, 
                self.frame_number, 
                self.conf, 
                max_association_distance_px
            )
        elif self.conf['tracking_method'] == 'kalman':
            if not self.object_filter:
                # TODO: figure out why even with a small association distance, the bboxes still jump
                metric = NearestNeighborDistanceMetric("euclidean", self.mm_to_px(self.conf["max_association_dist_mm"]))                
                self.object_filter = kalman.Tracker(metric,self.conf)
            kalman.filter_detections(detections, self.conf, self.object_filter)
            return kalman.tracks_to_object_history(self.object_filter.tracks, object_history, self.frame_number)
        else:
            raise ValueError(f"Invalid tracking method: {self.conf['tracking_method']}")

    def update_buffers_calculate_means(self, img):
        if self.framebuffer is None:
            self.framebuffer = img[:, :, np.newaxis]
        else:
            self.framebuffer = np.concatenate(
                (img[..., np.newaxis], self.framebuffer), axis=2
            )

        # Once the buffer is full+1, delete the last frame and calculate the means
        if self.framebuffer.shape[2] > self.conf["short_mean_frames"]:
            if self.short_mean_float is None:
                self.short_mean_float = np.mean(
                    self.framebuffer[:, :, : self.conf["short_mean_frames"]], axis=2
                ).astype("float64")
            else:
                short_mean_change = (
                    1.0
                    / self.conf["short_mean_frames"]
                    * (
                        self.framebuffer[:, :, 0].astype("float64")
                        - self.framebuffer[:, :, self.conf["short_mean_frames"]].astype(
                            "float64"
                        )
                    )
                )
                self.short_mean_float = self.short_mean_float + short_mean_change
            self.framebuffer = self.framebuffer[:, :, : self.conf["short_mean_frames"]]

            # If there is no current mean_buffer, initialize it with the current mean
            if self.mean_buffer is None:
                self.mean_buffer = self.short_mean_float.astype("uint8")[
                    :, :, np.newaxis
                ]
                self.mean_buffer_counter = 1

            # else if another conf["short_mean_frames"] number of frames have passed, add the current_mean
            elif self.mean_buffer_counter % self.conf["short_mean_frames"] == 0:
                self.mean_buffer = np.concatenate(
                    (
                        self.short_mean_float.astype("uint8")[..., np.newaxis],
                        self.mean_buffer,
                    ),
                    axis=2,
                )

                # once the long mean buffer is full+1, take the end off, and calculate the new long_mean
                mean_buffer_length = int(
                    self.conf["long_mean_frames"] / self.conf["short_mean_frames"]
                )
                if self.mean_buffer.shape[2] > mean_buffer_length:
                    if self.long_mean_float is None:
                        self.long_mean_float = np.mean(
                            self.mean_buffer[:, :, :mean_buffer_length], axis=2
                        ).astype("float64")
                    else:
                        long_mean_change = (
                            1.0
                            / mean_buffer_length
                            * (
                                self.mean_buffer[:, :, 0].astype("float64")
                                - self.mean_buffer[:, :, mean_buffer_length].astype(
                                    "float64"
                                )
                            )
                        )
                        self.long_mean_float = self.long_mean_float + long_mean_change

                    # Delete the oldest mean in the buffer
                    self.mean_buffer = self.mean_buffer[
                        :,
                        :,
                        :mean_buffer_length,
                    ]
                self.mean_buffer_counter = 1

            else:
                self.mean_buffer_counter += 1

    def mask_regions(self, img, area="sonar_controls"):
        if area == "non_object_space":
            if img.shape[:1] != self.non_object_space_mask.shape[:1]:
                percent_difference = (
                    img.shape[0] / self.non_object_space_mask.shape[0] * 100
                )

                np.place(
                    img,
                    resize_img(self.non_object_space_mask, percent_difference) < 100,
                    0,
                )
            else:
                np.place(img, self.non_object_space_mask < 100, 0)
        elif area == "sonar_controls":
            if img.shape[:1] != self.sonar_controls_mask.shape[:1]:
                percent_difference = (
                    img.shape[0] / self.sonar_controls_mask.shape[0] * 100
                )

                np.place(
                    img,
                    resize_img(self.sonar_controls_mask, percent_difference) < 100,
                    0,
                )
            else:
                np.place(img, self.sonar_controls_mask < 100, 0)
        return img

    @staticmethod
    def rgb_to_gray(img):
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    @staticmethod
    def ceil_to_odd_int(number):
        number = int(np.ceil(number))
        return number + 1 if number % 2 == 0 else number

    def mm_to_px(self, millimeters):
        px = (
            millimeters
            * self.conf["input_pixels_per_mm"]
            * self.conf["downsample"]
            / 100
        )
        return px

    def classify_detections(self, df):
        rotated_velocities = rotate_velocity_vectors(df[["v_x", "v_y"]], conf=self.conf)
        df = pd.concat([df, rotated_velocities], axis=1)

        abs_vel = np.linalg.norm(self.conf["river_pixel_velocity"])
        df.classification = ""
        for ID in df.ID.unique():
            obj = df.loc[df.ID == ID]
            if obj.shape[0] < np.max([20, 10]):
                df.loc[df.ID == ID, "classification"] = "object"

            if (
                abs(obj.v_yr).max()
                > self.conf["deviation_from_river_velocity"] * abs_vel
            ):
                df.loc[df.ID == ID, "classification"] = "fish"
            elif (
                abs(obj.v_xr - abs_vel).max()
                > self.conf["deviation_from_river_velocity"] * abs_vel
            ):
                df.loc[df.ID == ID, "classification"] = "fish"
            else:
                df.loc[df.ID == ID, "classification"] = "object"

        return df
