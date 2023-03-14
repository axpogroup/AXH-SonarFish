import os

import cv2 as cv
import numpy as np
from algorithm.DetectedObject import DetectedObject
from algorithm.utils import get_elapsed_ms, resize_img


class FishDetector:
    def __init__(self, settings_dict):
        self.conf = settings_dict
        self.frame_number = 0
        self.latest_obj_index = 0
        self.current_objects = {}

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
        self.short_mean = None
        self.long_mean = None

    def process_frame(self, raw_frame, object_history):
        start = cv.getTickCount()
        runtimes_ms = {}
        frames = {}
        self.current_objects = object_history
        frames["raw"] = raw_frame

        # Image enhancement
        if self.conf["downsample"]:
            frames["raw_downsampled"] = resize_img(raw_frame, self.conf["downsample"])
            frames["gray"] = self.rgb_to_gray(frames["raw_downsampled"])
        else:
            frames["gray"] = self.rgb_to_gray(frames["raw"])

        enhanced_temp = self.mask_regions(frames["gray"], area="sonar_controls")
        enhanced_temp = cv.convertScaleAbs(
            enhanced_temp, alpha=self.conf["contrast"], beta=self.conf["brightness"]
        )
        frames["gray_boosted"] = enhanced_temp
        self.update_buffer(enhanced_temp)

        if self.frame_number < self.conf["long_mean_frames"]:
            runtimes_ms["enhance"] = get_elapsed_ms(start)
            runtimes_ms["detection_tracking"] = (
                get_elapsed_ms(start) - runtimes_ms["enhance"]
            )
        else:
            enhanced_temp = self.calc_difference_from_buffer()
            frames["long_mean"] = self.long_mean
            frames["short_mean"] = self.short_mean
            frames["difference"] = (enhanced_temp + 127).astype("uint8")
            frames["absolute_difference"] = (abs(enhanced_temp) + 127).astype("uint8")
            # TOD: validate if the blur helps
            adaptive_threshold = self.conf["difference_threshold_scaler"] * cv.blur(
                self.long_mean, (10, 10)
            )
            enhanced_temp[abs(enhanced_temp) < adaptive_threshold] = 0
            frames["difference_thresholded"] = (enhanced_temp + 127).astype("uint8")

            # enhanced_temp = cv.Laplacian(enhanced_temp, cv.CV_64F, ksize=5)
            # enhanced_temp = (enhanced_temp + 127)
            # enhanced_temp[enhanced_temp < 0] = 0
            # enhanced_temp[enhanced_temp > 255] = 255
            enhanced_temp = (abs(enhanced_temp) + 127).astype("uint8")
            median_filter_kernel_px = self.mm_to_px(
                self.conf["median_filter_kernel_mm"], uneven=True
            )
            enhanced_temp = cv.medianBlur(enhanced_temp, median_filter_kernel_px)
            frames["median_filter"] = enhanced_temp
            runtimes_ms["enhance"] = get_elapsed_ms(start)

            # Detection and Tracking
            detections, frames = self.find_points_of_interest(enhanced_temp, frames)
            object_history = self.associate_detections(detections, object_history)
            runtimes_ms["detection_tracking"] = (
                get_elapsed_ms(start) - runtimes_ms["enhance"]
            )

        runtimes_ms["total"] = get_elapsed_ms(start)
        self.frame_number += 1
        return frames, object_history, runtimes_ms

    # TOD: This function will be updated once the algorithm development resumes
    def associate_detections(self, detections, object_history):
        if len(object_history) == 0:
            object_history = detections
            return object_history

        object_midpoints = [
            existing_object.midpoints[-1]
            for _, existing_object in object_history.items()
            if (
                self.frame_number - existing_object.frames_observed[-1]
                < self.conf["phase_out_after_x_frames"]
            )
        ]

        if len(object_midpoints) == 0:
            object_history = detections
            return object_history

        object_ids = [
            key
            for key, existing_object in object_history.items()
            if (
                self.frame_number - existing_object.frames_observed[-1]
                < self.conf["phase_out_after_x_frames"]
            )
        ]

        new_objects = []
        associations = {}
        for _, detection in detections.items():
            min_id, min_dist = self.closest_point(
                detection.midpoints[-1], object_midpoints
            )
            max_association_distance_px = self.mm_to_px(
                self.conf["max_association_dist_mm"]
            )
            if min_dist < max_association_distance_px:
                if object_ids[min_id] in associations.keys():
                    if associations[object_ids[min_id]]["distance"] > min_dist:
                        new_objects.append(
                            associations[object_ids[min_id]]["detection"]
                        )
                        associations[object_ids[min_id]] = {
                            "detection_id": detection.ID,
                            "distance": min_dist,
                            "detection": detection,
                        }
                    new_objects.append(detection)
                else:
                    associations[object_ids[min_id]] = {
                        "detection_id": detection.ID,
                        "distance": min_dist,
                        "detection": detection,
                    }
            else:
                new_objects.append(detection)

        for new_object in new_objects:
            object_history[new_object.ID] = new_object

        for existing_object_id, associated_detection in associations.items():
            object_history[existing_object_id].update_object(
                detections[associated_detection["detection_id"]]
            )

        return object_history

    @staticmethod
    def closest_point(point, points):
        points = np.asarray(points)
        dist_2 = np.sqrt(np.sum((points - point) ** 2, axis=1))
        min_index = np.argmin(dist_2)
        return min_index, dist_2[min_index]

    # TOD: This function will be updated once the algorithm development resumes
    def find_points_of_interest(self, enhanced_frame, frame_dict):
        # Make positive and negative differences the same
        enhanced_frame = (abs(enhanced_frame.astype("int16") - 127) + 127).astype(
            "uint8"
        )

        # Threshold
        ret, thres = cv.threshold(
            enhanced_frame, 127 + self.conf["difference_threshold_scaler"], 255, 0
        )
        frame_dict["binary"] = thres
        # Alternative consolidation - dilate
        dilation_kernel_px = self.mm_to_px(self.conf["dilation_kernel_mm"], uneven=True)
        kernel = cv.getStructuringElement(
            cv.MORPH_ELLIPSE, (dilation_kernel_px, dilation_kernel_px)
        )
        frame_dict["dilated"] = cv.dilate(thres, kernel, iterations=1)
        # frame_dict["closed"] = cv.morphologyEx(thres, cv.MORPH_CLOSE, kernel)
        # frame_dict["opened"] = cv.morphologyEx(thres, cv.MORPH_OPEN, kernel)
        # img = self.spatial_filter(img, kernel_size=15, method='median')

        thres = frame_dict["dilated"]
        contours, hier = cv.findContours(
            thres, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        detections = {}
        for contour in contours:
            new_object = DetectedObject(
                self.get_new_id(), contour, self.frame_number, self.conf
            )
            detections[new_object.ID] = new_object

        return detections, frame_dict

    def update_buffer(self, img):
        if self.framebuffer is None:
            self.framebuffer = img[:, :, np.newaxis]
        else:
            self.framebuffer = np.concatenate(
                (img[..., np.newaxis], self.framebuffer), axis=2
            )

        if self.framebuffer.shape[2] > self.conf["short_mean_frames"]:
            self.framebuffer = self.framebuffer[:, :, : self.conf["short_mean_frames"]]

    def calc_difference_from_buffer(self):
        self.short_mean = np.mean(
            self.framebuffer[:, :, : self.conf["short_mean_frames"]], axis=2
        ).astype("uint8")

        # If there is no current mean_buffer, initialize it with the current mean
        if self.mean_buffer is None:
            self.mean_buffer = self.short_mean[:, :, np.newaxis]
            self.mean_buffer_counter = 1
            self.long_mean = np.mean(self.mean_buffer, axis=2).astype("uint8")

        # else if another conf["short_mean_frames"] number of frames have passed, add the current_mean
        elif self.mean_buffer_counter % self.conf["short_mean_frames"] == 0:
            self.mean_buffer = np.concatenate(
                (self.short_mean[..., np.newaxis], self.mean_buffer), axis=2
            )
            # if the long mean buffer has gotten to big, take the end off
            if self.mean_buffer.shape[2] > int(
                self.conf["long_mean_frames"] / self.conf["short_mean_frames"]
            ):
                self.mean_buffer = self.mean_buffer[
                    :,
                    :,
                    : int(
                        self.conf["long_mean_frames"] / self.conf["short_mean_frames"]
                    ),
                ]

            self.long_mean = np.mean(self.mean_buffer, axis=2).astype("uint8")
            self.mean_buffer_counter = 1
            # else:
            #     self.mean_buffer_counter += 1
        else:
            self.mean_buffer_counter += 1

        return self.short_mean.astype("int16") - self.long_mean.astype("int16")

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

    def get_new_id(self):
        if self.latest_obj_index > 300000:
            self.latest_obj_index = 0
        self.latest_obj_index += 1
        return self.latest_obj_index

    def mm_to_px(self, millimeters, uneven=False):
        px = (
            millimeters
            * self.conf["input_pixels_per_mm"]
            * self.conf["downsample"]
            / 100
        )
        if uneven:
            px = int(px)
            if (float(px) / 2 % 1) == 0:
                px += 1
        return px
