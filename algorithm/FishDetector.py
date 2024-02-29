from pathlib import Path
from typing import Dict

import cv2 as cv
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import scipy.cluster.vq as scv

from algorithm.DetectedObject import DetectedBlob, KalmanTrackedBlob
from algorithm.flow_conditions import rotate_velocity_vectors
from algorithm.matching.distance import DistanceMetric
from algorithm.tracking_filters import kalman, nearest_neighbor
from algorithm.utils import get_elapsed_ms, resize_img


class FishDetector:
    def __init__(self, settings_dict):
        self.object_filter = None
        self.conf = settings_dict
        self.frame_number = 0
        self.latest_obj_index = 0

        # Enhancement
        self.framebuffer = None
        self.mean_buffer = None
        self.mean_buffer_counter = None
        self.short_mean_float = None
        self.long_mean_float = None

    def detect_objects(self, raw_frame) -> tuple[Dict[int, DetectedBlob], dict, dict]:
        start = cv.getTickCount()
        runtimes_ms = {}
        frame_dict = {"raw": raw_frame}
        self.enhance_image(frame_dict)
        self.frame_number += 1
        if self.long_mean_float is None:
            runtimes_ms["enhance"] = get_elapsed_ms(start)
            runtimes_ms["detection_tracking"] = get_elapsed_ms(start) - runtimes_ms["enhance"]
            runtimes_ms["total"] = get_elapsed_ms(start)

            return {}, frame_dict, runtimes_ms
        else:
            enhanced_temp = (self.short_mean_float - self.long_mean_float).astype("int16")
            frame_dict["long_mean"] = self.long_mean_float.astype("uint8")
            frame_dict["short_mean"] = self.short_mean_float.astype("uint8")
            frame_dict["difference"] = (enhanced_temp + 127).astype("uint8")
            frame_dict["absolute_difference"] = (abs(enhanced_temp) + 127).astype("uint8")

            adaptive_threshold = self.conf["difference_threshold_scaler"] * cv.blur(
                self.long_mean_float.astype("uint8"), (10, 10)
            )
            enhanced_temp[abs(enhanced_temp) < adaptive_threshold] = 0
            frame_dict["difference_thresholded"] = (enhanced_temp + 127).astype("uint8")

            enhanced_temp = (abs(enhanced_temp) + 127).astype("uint8")
            frame_dict["difference_thresholded_abs"] = enhanced_temp
            median_filter_kernel_px = self.mm_to_px(self.conf["median_filter_kernel_mm"])
            enhanced_temp = cv.medianBlur(enhanced_temp, self.ceil_to_odd_int(median_filter_kernel_px))
            frame_dict["median_filter"] = enhanced_temp
            runtimes_ms["enhance"] = get_elapsed_ms(start)

            # Threshold to binary
            ret, thres = cv.threshold(enhanced_temp, 127 + self.conf["difference_threshold_scaler"], 255, 0)
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

            detections = self.extract_keypoints(frame_dict)

            runtimes_ms["detection_tracking"] = get_elapsed_ms(start) - runtimes_ms["enhance"]

        runtimes_ms["total"] = get_elapsed_ms(start)
        return detections, frame_dict, runtimes_ms

    def enhance_image(self, frame_dict):
        # Image enhancement
        if self.conf.get("downsample"):
            frame_dict["raw_downsampled"] = resize_img(frame_dict["raw"], self.conf["downsample"])
            frame_dict["gray"] = self.rgb_to_gray(frame_dict["raw_downsampled"])
        else:
            frame_dict["gray"] = self.rgb_to_gray(frame_dict["raw"])
        enhanced_temp = self.mask_regions(frame_dict["gray"], self.conf["mask_file"])
        enhanced_temp = cv.convertScaleAbs(enhanced_temp, alpha=self.conf["contrast"], beta=self.conf["brightness"])
        self.update_buffers_calculate_means(enhanced_temp)
        frame_dict["gray_boosted"] = enhanced_temp
        return frame_dict

    def extract_keypoints(self, frame_dict) -> Dict[int, DetectedBlob]:
        # Extract keypoints
        contours, _ = cv.findContours(frame_dict["dilated"], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        detections: list[DetectedBlob] = {}
        for contour in contours:
            new_object = DetectedBlob(
                identifier=self.latest_obj_index,
                frame_number=self.frame_number,
                contour=contour,
                frame=frame_dict,
            )
            detections[new_object.ID] = new_object
            self.latest_obj_index += 1
        return detections

    def associate_detections(
        self,
        detections: dict[int, DetectedBlob],
        object_history: dict[int, KalmanTrackedBlob],
        processed_frame_dict,
    ) -> Dict[int, KalmanTrackedBlob]:
        if len(detections) == 0:
            return object_history

        if self.conf["tracking_method"] == "nearest_neighbor":
            return nearest_neighbor.associate_detections(
                detections,
                object_history,
                self.frame_number,
                self.conf,
                self.max_association_distance_px,
            )
        elif self.conf["tracking_method"] == "kalman":
            primary_metric = DistanceMetric(
                self.conf["filter_blob_matching_metric"],
                self.kf_metric_matching_thresh,
                budget=self.conf["kalman_trace_history_matching_budget"],
            )
            if "filter_blob_elimination_metric" in self.conf:
                elimination_metric = DistanceMetric(
                    self.conf["filter_blob_elimination_metric"],
                    self.max_association_distance_px,
                )
            else:
                elimination_metric = None

            if not self.object_filter:
                self.object_filter = kalman.Tracker(primary_metric, elimination_metric, self.conf)

            kalman.filter_detections(detections, self.object_filter)
            return kalman.tracks_to_object_history(
                object_filter=self.object_filter,
                object_history=object_history,
                frame_number=self.frame_number,
                processed_frame_dict=processed_frame_dict,
                bbox_size_to_stddev_ratio_threshold=self.conf.get("bbox_size_to_stddev_ratio_threshold"),
            )
        else:
            raise ValueError(f"Invalid tracking method: {self.conf['tracking_method']}")

    def update_buffers_calculate_means(self, img):
        if self.framebuffer is None:
            self.framebuffer = img[:, :, np.newaxis]
        else:
            self.framebuffer = np.concatenate((img[..., np.newaxis], self.framebuffer), axis=2)

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
                        - self.framebuffer[:, :, self.conf["short_mean_frames"]].astype("float64")
                    )
                )
                self.short_mean_float = self.short_mean_float + short_mean_change
            self.framebuffer = self.framebuffer[:, :, : self.conf["short_mean_frames"]]

            # If there is no current mean_buffer, initialize it with the current mean
            if self.mean_buffer is None:
                self.mean_buffer = self.short_mean_float.astype("uint8")[:, :, np.newaxis]
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
                mean_buffer_length = int(self.conf["long_mean_frames"] / self.conf["short_mean_frames"])
                if self.mean_buffer.shape[2] > mean_buffer_length:
                    if self.long_mean_float is None:
                        self.long_mean_float = np.mean(self.mean_buffer[:, :, :mean_buffer_length], axis=2).astype(
                            "float64"
                        )
                    else:
                        long_mean_change = (
                            1.0
                            / mean_buffer_length
                            * (
                                self.mean_buffer[:, :, 0].astype("float64")
                                - self.mean_buffer[:, :, mean_buffer_length].astype("float64")
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

    def mask_regions(self, img, mask_file="sonar_controls.png"):

        mask = cv.imread(
            (Path(self.conf["mask_directory"]) / mask_file).as_posix(),
            cv.IMREAD_GRAYSCALE,
        )

        if img.shape[:1] != mask.shape[:1]:
            percent_difference = img.shape[0] / mask.shape[0] * 100

            np.place(
                img,
                resize_img(mask, percent_difference) < 100,
                0,
            )
        else:
            np.place(img, mask < 100, 0)

        return img

    def rgb_to_gray(self, img):
        if self.conf["video_colormap"] == "red":
            return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        elif self.conf["video_colormap"] == "jet":
            return colormap_to_array(img)
        else:
            raise ValueError(f"Invalid colormap: {self.settings_dict['video_colormap']}, must be 'red' or 'jet'")

    @staticmethod
    def ceil_to_odd_int(number):
        number = int(np.ceil(number))
        return number + 1 if number % 2 == 0 else number

    def mm_to_px(self, millimeters):
        px = millimeters * self.conf["input_pixels_per_mm"] * self.conf["downsample"] / 100
        return px

    def classify_detections(self, df):
        rotated_velocities = rotate_velocity_vectors(df[["v_x", "v_y"]], conf=self.conf)
        df = pd.concat([df, rotated_velocities], axis=1)

        abs_vel = np.linalg.norm(self.conf["river_pixel_velocity"])
        df.classification = ""
        for ID in df.id.unique():
            obj = df.loc[df.id == ID]
            if obj.shape[0] < np.max([20, 10]):
                df.loc[df.id == ID, "classification"] = "object"

            if abs(obj.v_yr).max() > self.conf["deviation_from_river_velocity"] * abs_vel:
                df.loc[df.id == ID, "classification"] = "fish"
            elif abs(obj.v_xr - abs_vel).max() > self.conf["deviation_from_river_velocity"] * abs_vel:
                df.loc[df.id == ID, "classification"] = "fish"
            else:
                df.loc[df.id == ID, "classification"] = "object"

        return df

    @property
    def max_association_distance_px(self):
        return self.mm_to_px(self.conf["max_association_dist_mm"])

    @property
    def kf_metric_matching_thresh(self):
        if self.conf["filter_blob_matching_metric"] == "euclidean_distance":
            matching_threshold = self.max_association_distance_px
        else:
            matching_threshold = self.conf["filter_association_thresh"]
        return matching_threshold


def colormap_to_array(input_array, colormap=cm.jet):
    # http://stackoverflow.com/questions/3720840/how-to-reverse-color-map-image-to-scalar-values/3722674#3722674
    colormap_mapping = colormap(np.linspace(0.0, 1.0, 255))[:, 0:3]

    # We need to reverse color channels since opencv uses BGR and not RGB
    reshaped_array = input_array[:, :, ::-1].reshape(
        (input_array.shape[0] * input_array.shape[1], input_array.shape[2])
    )
    color_nearest_neighbor, _ = scv.vq(reshaped_array, colormap_mapping)
    scaled_values = color_nearest_neighbor.astype("uint8")
    reshaped_values = scaled_values.reshape(input_array.shape[0], input_array.shape[1])

    return reshaped_values
