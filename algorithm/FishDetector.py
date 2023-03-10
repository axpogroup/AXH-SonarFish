import os

import cv2 as cv
import numpy as np
from DetectedObject import DetectedObject
from utils import get_elapsed_ms, resize_img


class FishDetector:
    def __init__(self, settings_dict):
        self.settings_dict = settings_dict
        self.frame_number = 0
        self.latest_obj_index = 0
        self.current_objects = {}

        # Masks
        self.non_object_space_mask = cv.imread(
            os.path.join(settings_dict["mask_directory"], "fish.png"),
            cv.IMREAD_GRAYSCALE,
        )
        self.sonar_controls_mask = cv.imread(
            os.path.join(settings_dict["mask_directory"], "full.png"),
            cv.IMREAD_GRAYSCALE,
        )

        # Enhancement
        self.framebuffer = None
        self.mean_buffer = None
        self.mean_buffer_counter = None
        self.short_mean = None
        self.long_mean = None

        self.downsample = settings_dict["downsample"]
        self.contrast = settings_dict["contrast"]
        self.brightness = settings_dict["brightness"]
        self.long_mean_frames = settings_dict["long_mean_frames"]
        self.current_mean_frames = settings_dict["short_mean_frames"]
        self.median_filter_kernel = settings_dict["median_filter_kernel"]
        self.dilatation_kernel = settings_dict["dilatation_kernel"]
        self.difference_threshold_scaler = settings_dict["difference_threshold_scaler"]

        # Detection and Tracking
        self.max_association_dist = settings_dict["max_association_dist"]
        self.phase_out_after_x_frames = settings_dict["phase_out_after_x_frames"]
        self.min_occurences_in_last_x_frames = settings_dict[
            "min_occurences_in_last_x_frames"
        ]

    def process_frame(self, raw_frame, object_history):
        start = cv.getTickCount()
        runtimes_ms = {}
        frames = {}
        self.current_objects = object_history
        frames["raw"] = raw_frame

        # Image enhancement
        if self.downsample:
            frames["raw_downsampled"] = resize_img(raw_frame, self.downsample)
            frames["gray"] = self.rgb_to_gray(frames["raw_downsampled"])
        else:
            frames["gray"] = self.rgb_to_gray(frames["raw"])

        enhanced_temp = self.mask_regions(frames["gray"], area="sonar_controls")
        enhanced_temp = cv.convertScaleAbs(
            enhanced_temp, alpha=self.contrast, beta=self.brightness
        )
        frames["gray_boosted"] = enhanced_temp
        self.update_buffer(enhanced_temp)

        if self.frame_number < self.long_mean_frames:
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
            adaptive_threshold = self.difference_threshold_scaler * cv.blur(
                self.long_mean, (10, 10)
            )
            enhanced_temp[abs(enhanced_temp) < adaptive_threshold] = 0
            frames["difference_thresholded"] = (enhanced_temp + 127).astype("uint8")

            # enhanced_temp = cv.Laplacian(enhanced_temp, cv.CV_64F, ksize=5)
            # enhanced_temp = (enhanced_temp + 127)
            # enhanced_temp[enhanced_temp < 0] = 0
            # enhanced_temp[enhanced_temp > 255] = 255
            enhanced_temp = (abs(enhanced_temp) + 127).astype("uint8")
            enhanced_temp = cv.medianBlur(enhanced_temp, self.median_filter_kernel)
            frames["median_filter"] = enhanced_temp
            runtimes_ms["enhance"] = get_elapsed_ms(start)

            # Detection and Tracking
            detections, frames = self.find_points_of_interest(
                enhanced_temp, frames, mode="contour"
            )
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
                < self.phase_out_after_x_frames
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
                < self.phase_out_after_x_frames
            )
        ]

        new_objects = []
        associations = {}
        for _, detection in detections.items():
            min_id, min_dist = self.closest_point(
                detection.midpoints[-1], object_midpoints
            )
            if min_dist < self.max_association_dist:
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
    def find_points_of_interest(self, enhanced_frame, frame_dict, mode="contour"):
        if mode == "contour":
            # Make positive and negative differences the same
            enhanced_frame = (abs(enhanced_frame.astype("int16") - 127) + 127).astype(
                "uint8"
            )

            # Threshold
            ret, thres = cv.threshold(
                enhanced_frame, 127 + self.difference_threshold_scaler, 255, 0
            )
            frame_dict["binary"] = thres
            # Alternative consolidation - dilate
            # kernel = np.ones((self.dilatation_kernel, self.dilatation_kernel), "uint8")
            kernel = cv.getStructuringElement(
                cv.MORPH_ELLIPSE, (self.dilatation_kernel, self.dilatation_kernel)
            )
            frame_dict["dilated"] = cv.dilate(thres, kernel, iterations=1)
            frame_dict["closed"] = cv.morphologyEx(thres, cv.MORPH_CLOSE, kernel)
            frame_dict["opened"] = cv.morphologyEx(thres, cv.MORPH_OPEN, kernel)
            # img = self.spatial_filter(img, kernel_size=15, method='median')

            thres = frame_dict["dilated"]
            contours, hier = cv.findContours(
                thres, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )
            detections = {}
            for contour in contours:
                new_object = DetectedObject(
                    self.get_new_id(), contour, self.frame_number, self.settings_dict
                )
                detections[new_object.ID] = new_object

            return detections, frame_dict

        elif mode == "blob":  # TOD0 there is an issue, it Segmentation faults instantly
            blob_detector = cv.SimpleBlobDetector_create()
            keypoints = blob_detector.detect(enhanced_frame)
            # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
            im_with_keypoints = cv.drawKeypoints(
                enhanced_frame,
                keypoints,
                enhanced_frame,
                (0, 0, 255),
                cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )
            frame_dict["blobs"] = im_with_keypoints
            return {}, frame_dict

    def update_buffer(self, img):
        if self.framebuffer is None:
            self.framebuffer = img[:, :, np.newaxis]
        else:
            self.framebuffer = np.concatenate(
                (img[..., np.newaxis], self.framebuffer), axis=2
            )

        if self.framebuffer.shape[2] > self.current_mean_frames:
            self.framebuffer = self.framebuffer[:, :, : self.current_mean_frames]

    def calc_difference_from_buffer(self):
        self.short_mean = np.mean(
            self.framebuffer[:, :, : self.current_mean_frames], axis=2
        ).astype("uint8")

        # If there is no current mean_buffer, initialize it with the current mean
        if self.mean_buffer is None:
            self.mean_buffer = self.short_mean[:, :, np.newaxis]
            self.mean_buffer_counter = 1
            self.long_mean = np.mean(self.mean_buffer, axis=2).astype("uint8")

        # else if another current_mean_frames number of frames have passed, add the current_mean
        elif self.mean_buffer_counter % self.current_mean_frames == 0:
            self.mean_buffer = np.concatenate(
                (self.short_mean[..., np.newaxis], self.mean_buffer), axis=2
            )
            # if the long mean buffer has gotten to big, take the end off
            if self.mean_buffer.shape[2] > int(
                self.long_mean_frames / self.current_mean_frames
            ):
                self.mean_buffer = self.mean_buffer[
                    :, :, : int(self.long_mean_frames / self.current_mean_frames)
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
