from pathlib import Path
from typing import Dict, Optional

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
from algorithm.settings import Settings



class FishDetector:
    def __init__(self, settings:Settings, init_detector: Optional["FishDetector"] = None):
        self.object_filter = None
        self.__settings = settings
        self.frame_number = 0
        self.latest_obj_index = 0

        # Enhancement
        if init_detector:
            self.framebuffer = init_detector.framebuffer
            self.mean_buffer = init_detector.mean_buffer
            self.mean_buffer_counter = init_detector.mean_buffer_counter
            self.burn_in_video_name = init_detector.__settings.file_name
        else:
            self.framebuffer = None
            self.mean_buffer = None
            self.mean_buffer_counter = None
            self.burn_in_video_name = None
        self.short_mean_float = None
        self.long_mean_float = None
        mask_file = self.__settings.mask_file or "sonar_controls.png"
        self.mask = cv.imread(
            (Path(self.__settings.mask_directory) / (mask_file)).as_posix(),
            cv.IMREAD_GRAYSCALE,
        )
        assert self.mask is not None, f"Mask file not found: {mask_file}"

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
            frame_dict["difference"] = (enhanced_temp + 127).astype("uint8")
            frame_dict["long_mean"] = self.long_mean_float.astype("uint8")
            frame_dict["short_mean"] = self.short_mean_float.astype("uint8")
            adaptive_threshold = self.__settings.difference_threshold_scaler * cv.blur(
                self.long_mean_float.astype("uint8"), (10, 10)
            )
            enhanced_temp[abs(enhanced_temp) < adaptive_threshold] = 0
            frame_dict["difference_thresholded"] = (enhanced_temp + 127).astype("uint8")
            enhanced_temp = (abs(enhanced_temp) + 127).astype("uint8")
            median_filter_kernel_px = self.mm_to_px(self.__settings.median_filter_kernel_mm)
            enhanced_temp = cv.medianBlur(enhanced_temp, self.ceil_to_odd_int(median_filter_kernel_px))
            frame_dict["median_filter"] = enhanced_temp
            runtimes_ms["enhance"] = get_elapsed_ms(start)
            # Threshold to binary
            ret, thres = cv.threshold(enhanced_temp, 127 + self.__settings.difference_threshold_scaler, 255, 0)
            self.dilate_frame(frame_dict, thres)
            detections = self.extract_keypoints(frame_dict)
            runtimes_ms["detection_tracking"] = get_elapsed_ms(start) - runtimes_ms["enhance"]
        runtimes_ms["total"] = get_elapsed_ms(start)
        return detections, frame_dict, runtimes_ms

        

    def dilate_frame(self, frame_dict, thres):
        dilation_kernel_px = self.mm_to_px(self.__settings.dilation_kernel_mm)
        kernel = cv.getStructuringElement(
            cv.MORPH_ELLIPSE,
            (
                self.ceil_to_odd_int(dilation_kernel_px),
                self.ceil_to_odd_int(dilation_kernel_px),
            ),
        )
        frame_dict["dilated"] = cv.dilate(thres, kernel, iterations=1)

    def enhance_image(self, frame_dict):
        if self.__settings.downsample:
            frame_dict["raw_downsampled"] = resize_img(frame_dict["raw"], self.__settings.downsample)
            frame_dict["gray"] = self.rgb_to_gray(frame_dict["raw_downsampled"])
        else:
            frame_dict["gray"] = self.rgb_to_gray(frame_dict["raw"])
        enhanced_temp = self.mask_regions(frame_dict["gray"])
        enhanced_temp = cv.convertScaleAbs(enhanced_temp, alpha=self.__settings.contrast, beta=self.__settings.brightness)
        self.update_buffers_calculate_means(enhanced_temp)
        frame_dict["gray_boosted"] = enhanced_temp
        return frame_dict

    def extract_keypoints(self, frame_dict) -> Dict[int, DetectedBlob]:
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

        if self.__settings.tracking_method == "nearest_neighbor":
            return nearest_neighbor.associate_detections(
                detections,
                object_history,
                self.frame_number,
                self.__settings,
                self.max_association_distance_px,
            )
        elif self.__settings.tracking_method == "kalman":
            primary_metric = DistanceMetric(
                self.__settings.filter_blob_matching_metric,
                self.kf_metric_matching_thresh,
                budget=self.__settings.kalman_trace_history_matching_budget,
            )
            if "filter_blob_elimination_metric" in self.__settings:
                elimination_metric = DistanceMetric(
                    self.__settings.filter_blob_elimination_metric,
                    self.max_association_distance_px,
                )
            else:
                elimination_metric = None

            if not self.object_filter:
                self.object_filter = kalman.Tracker(self.__settings,primary_metric, elimination_metric )

            kalman.filter_detections(detections, self.object_filter)
            return kalman.tracks_to_object_history(
                object_filter=self.object_filter,
                object_history=object_history,
                frame_number=self.frame_number,
                processed_frame_dict=processed_frame_dict,
                settings=self.__settings,
            )
        else:
            raise ValueError(f"Invalid tracking method: {self.__settings.tracking_method}")

    def update_buffers_calculate_means(self, img):
        if self.framebuffer is None:
            self.framebuffer = img[:, :, np.newaxis]
        else:
            self.framebuffer = np.concatenate((img[..., np.newaxis], self.framebuffer), axis=2)

        # Once the buffer is full+1, delete the last frame and calculate the means
        if self.framebuffer.shape[2] > self.__settings.short_mean_frames:
            if self.short_mean_float is None:
                self.short_mean_float = np.mean(
                    self.framebuffer[:, :, : self.__settings.short_mean_frames], axis=2
                ).astype("float64")
            else:
                short_mean_change = (
                    1.0
                    / self.__settings.short_mean_frames
                    * (
                        self.framebuffer[:, :, 0].astype("float64")
                        - self.framebuffer[:, :, self.__settings.short_mean_frames].astype("float64")
                    )
                )
                self.short_mean_float = self.short_mean_float + short_mean_change
            self.framebuffer = self.framebuffer[:, :, : self.__settings.short_mean_frames]

            # If there is no current mean_buffer, initialize it with the current mean
            if self.mean_buffer is None:
                self.mean_buffer = self.short_mean_float.astype("uint8")[:, :, np.newaxis]
                self.mean_buffer_counter = 1

            # else if another settings."short_mean_frames"] number of frames have passed, add the current_mean
            elif self.mean_buffer_counter % self.__settings.short_mean_frames == 0:
                self.mean_buffer = np.concatenate(
                    (
                        self.short_mean_float.astype("uint8")[..., np.newaxis],
                        self.mean_buffer,
                    ),
                    axis=2,
                )

                # once the long mean buffer is full+1, take the end off, and calculate the new long_mean
                mean_buffer_length = int(self.__settings.long_mean_frames / self.__settings.short_mean_frames)
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

    def mask_regions(self, img):
        if img.shape[:1] != self.mask.shape[:1]:
            percent_difference = img.shape[0] / self.mask.shape[0] * 100

            np.place(
                img,
                resize_img(self.mask, percent_difference) < 100,
                0,
            )
        else:
            np.place(img, self.mask < 100, 0)

        return img

    def rgb_to_gray(self, img):
        if self.__settings.video_colormap == "red":
            return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        elif self.__settings.video_colormap == "jet":
            return colormap_to_array(img)
        else:
            raise ValueError(f"Invalid colormap: {self.__settings.video_colormap}, must be 'red' or 'jet'")

    @staticmethod
    def ceil_to_odd_int(number):
        number = int(np.ceil(number))
        return number + 1 if number % 2 == 0 else number

    def mm_to_px(self, millimeters):
        px = millimeters * self.__settings.input_pixels_per_mm* self.__settings.downsample / 100
        return px

    def classify_detections(self, df):
        rotated_velocities = rotate_velocity_vectors(df[["v_x", "v_y"]])
        df = pd.concat([df, rotated_velocities], axis=1)

        abs_vel = np.linalg.norm(self.__settings.river_pixel_velocity)
        df.classification = ""
        for ID in df.id.unique():
            obj = df.loc[df.id == ID]
            if obj.shape[0] < np.max([20, 10]):
                df.loc[df.id == ID, "classification"] = "object"

            if abs(obj.v_yr).max() > self.__settings.deviation_from_river_velocity * abs_vel:
                df.loc[df.id == ID, "classification"] = "fish"
            elif abs(obj.v_xr - abs_vel).max() > self.__settings.deviation_from_river_velocity * abs_vel:
                df.loc[df.id == ID, "classification"] = "fish"
            else:
                df.loc[df.id == ID, "classification"] = "object"

        return df

    @property
    def max_association_distance_px(self):
        return self.mm_to_px(self.__settings.max_association_dist_mm)

    @property
    def kf_metric_matching_thresh(self):
        if self.__settings.filter_blob_matching_metric == "euclidean_distance":
            matching_threshold = self.max_association_distance_px
        else:
            matching_threshold = self.__settings.filter_association_thresh
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
