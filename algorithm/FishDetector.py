import os

import cv2 as cv
import numpy as np
from DetectedObject import DetectedObject


class FishDetector:
    def __init__(self, settings_dict):
        self.settings_dict = settings_dict
        self.frame_number = 0  # TOD Must not overflow - recycle
        self.total_runtime_ms = None
        self.current_raw = None
        self.current_raw_downsampled = None
        self.current_gray = None
        self.current_enhanced = None
        self.current_output = None
        self.current_classified = None

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
        self.long_mean = None
        self.long_std_dev = None
        self.current_mean = None
        self.enhance_time_ms = None
        self.current_long_mean_uint8 = None
        self.current_blurred_enhanced = None
        self.downsample = settings_dict["downsample"]
        self.long_mean_frames = settings_dict["long_mean_frames"]
        self.current_mean_frames = settings_dict["current_mean_frames"]
        self.std_dev_threshold = settings_dict["std_dev_threshold"]
        self.median_filter_kernel = settings_dict["median_filter_kernel"]
        self.blur_filter_kernel = settings_dict["blur_filter_kernel"]
        self.threshold_contours = settings_dict["threshold_contours"]

        # Detection and Tracking
        self.detections = {}
        self.current_objects = {}
        self.current_threshold = None
        self.associations = []
        self.latest_obj_index = 0
        self.detection_tracking_time_ms = None
        self.max_association_dist = settings_dict["max_association_dist"]
        self.phase_out_after_x_frames = settings_dict["phase_out_after_x_frames"]
        self.min_occurences_in_last_x_frames = settings_dict[
            "min_occurences_in_last_x_frames"
        ]

    def process_frame(self, raw_frame, secondary=None):
        start = cv.getTickCount()
        self.current_raw = raw_frame
        if self.downsample:
            self.current_raw_downsampled = self.resize_img(raw_frame, self.downsample)
            self.current_gray = self.rgb_to_gray(self.current_raw_downsampled)
        else:
            self.current_gray = self.rgb_to_gray(self.current_raw)
        self.current_enhanced = self.enhance_frame(self.current_gray)
        self.enhance_time_ms = int(
            (cv.getTickCount() - start) / cv.getTickFrequency() * 1000
        )

        if self.frame_number > self.long_mean_frames:
            self.detect_and_track(self.current_enhanced)
            self.detection_tracking_time_ms = (
                int((cv.getTickCount() - start) / cv.getTickFrequency() * 1000)
                - self.enhance_time_ms
            )

        self.frame_number += 1
        self.total_runtime_ms = int(
            (cv.getTickCount() - start) / cv.getTickFrequency() * 1000
        )
        return

    def enhance_frame(self, gray_frame):
        light = False  # TOD unsure if this still works
        enhanced_temp = self.mask_regions(gray_frame, area="non_object_space")
        if light:
            self.update_buffer_light(enhanced_temp)
            if self.frame_number < self.long_mean_frames:
                return enhanced_temp * 0
        else:
            self.update_buffer(enhanced_temp)
            if self.frame_number < self.long_mean_frames:
                return enhanced_temp * 0

        if light:
            enhanced_temp = self.calc_difference_from_buffer_light()
            enhanced_temp[abs(enhanced_temp) < 10] = 0
        else:
            enhanced_temp = self.calc_difference_from_buffer()
            enhanced_temp = self.threshold_diff(
                enhanced_temp, threshold=self.std_dev_threshold
            )

        self.current_long_mean_uint8 = self.long_mean.astype("uint8")

        # Transform into visual/uint8 image and filter salt and peper noise
        enhanced_temp = (enhanced_temp + 125).astype("uint8")
        enhanced_temp = cv.medianBlur(enhanced_temp, self.median_filter_kernel)
        return enhanced_temp

    def detect_and_track(self, enhanced_frame):
        self.detections = self.find_points_of_interest(enhanced_frame, mode="contour")
        self.current_objects = self.associate_detections(self.detections)
        self.current_objects = self.filter_objects(self.current_objects)
        return

    def filter_objects(self, current_objects):
        to_delete = []
        for ID, obj in current_objects.items():
            # Delete if it hasn't been observed in the last x frames
            if (
                self.frame_number - obj.frames_observed[-1]
                > self.phase_out_after_x_frames
            ):
                to_delete.append(ID)
                continue

            # Show if x occurences in the last y frames
            if (
                obj.occurences_in_last_x(
                    self.frame_number, self.min_occurences_in_last_x_frames[1]
                )
                >= self.min_occurences_in_last_x_frames[0]
            ):
                obj.show[-1] = True
            else:
                obj.show[-1] = False

        for key in to_delete:
            current_objects.pop(key)

        return current_objects

    def associate_detections(self, detections):
        if len(self.current_objects) == 0:
            self.current_objects = detections
            return self.current_objects

        object_midpoints = [
            existing_object.midpoints[-1]
            for _, existing_object in self.current_objects.items()
        ]
        object_ids = list(self.current_objects.keys())
        new_objects = []
        self.associations = []
        for detection_id, detection in detections.items():
            min_id, min_dist = self.closest_point(
                detection.midpoints[-1], object_midpoints
            )
            if min_dist < self.max_association_dist:
                self.associations.append(
                    {
                        "detection_id": detection.ID,
                        "existing_object_id": object_ids[min_id],
                        "distance": min_dist,
                    }
                )
            else:
                new_objects.append(detection)

        for new_object in new_objects:
            self.current_objects[new_object.ID] = new_object

        for association in self.associations:
            self.current_objects[association["existing_object_id"]].update_object(
                detections[association["detection_id"]]
            )

        return self.current_objects

    @staticmethod
    def closest_point(point, points):
        points = np.asarray(points)
        dist_2 = np.sqrt(np.sum((points - point) ** 2, axis=1))
        min_index = np.argmin(dist_2)
        return min_index, dist_2[min_index]

    def find_points_of_interest(self, enhanced_frame, mode="contour"):
        if mode == "contour":
            # Make positive and negative differences the same
            enhanced_frame = (abs(enhanced_frame.astype("int16") - 125) + 125).astype(
                "uint8"
            )

            # Consolidate the points
            enhanced_frame = cv.GaussianBlur(
                enhanced_frame,
                (self.blur_filter_kernel, self.blur_filter_kernel),
                0,
            )

            self.current_blurred_enhanced = enhanced_frame
            # Threshold
            ret, thres = cv.threshold(enhanced_frame, self.threshold_contours, 255, 0)
            self.current_threshold = thres
            # Alternative consolidation - dilate
            # kernel = np.ones((51, 51), 'uint8')
            # thres = cv.dilate(thres, kernel, iterations=1)
            # img = self.spatial_filter(img, kernel_size=15, method='median')

            contours, hier = cv.findContours(
                thres, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )
            detections = {}
            for contour in contours:
                new_object = DetectedObject(
                    self.get_new_id(), contour, self.frame_number, self.settings_dict
                )
                detections[new_object.ID] = new_object

            return detections

        elif mode == "blob":  # TOD there is an issue, it Segmentation faults instantly
            blob_detector = cv.SimpleBlobDetector()
            keypoints = blob_detector.detect(enhanced_frame)
            # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
            im_with_keypoints = cv.drawKeypoints(
                enhanced_frame,
                keypoints,
                enhanced_frame,
                (0, 0, 255),
                cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )
            return im_with_keypoints

    def update_buffer_light(self, img):
        if self.framebuffer is None:
            self.framebuffer = img[:, :, np.newaxis]
        else:
            self.framebuffer = np.concatenate(
                (img[..., np.newaxis], self.framebuffer), axis=2
            )

        if self.framebuffer.shape[2] > self.current_mean_frames:
            self.framebuffer = self.framebuffer[:, :, : self.current_mean_frames]

    def update_buffer(self, img):
        if self.framebuffer is None:
            self.framebuffer = img[:, :, np.newaxis]
        else:
            self.framebuffer = np.concatenate(
                (img[..., np.newaxis], self.framebuffer), axis=2
            )

        if self.framebuffer.shape[2] > self.long_mean_frames:
            self.framebuffer = self.framebuffer[:, :, : self.long_mean_frames]

    def calc_difference_from_buffer_light(self):
        self.current_mean = np.mean(self.framebuffer[:, :, :10], axis=2).astype("uint8")

        if self.mean_buffer is None:
            self.mean_buffer = self.current_mean[:, :, np.newaxis]
            self.mean_buffer_counter = 1
            self.long_mean = np.mean(self.mean_buffer, axis=2).astype("int16")
        elif self.mean_buffer_counter % self.current_mean_frames == 0:
            self.mean_buffer = np.concatenate(
                (self.current_mean[..., np.newaxis], self.mean_buffer), axis=2
            )
            if self.mean_buffer.shape[2] > int(
                self.long_mean_frames / self.current_mean_frames
            ):
                self.mean_buffer = self.mean_buffer[
                    :, :, : int(self.long_mean_frames / self.current_mean_frames)
                ]

            # if self.mean_buffer_counter % self.long_mean_frames == 0:
            self.long_mean = np.mean(self.mean_buffer, axis=2).astype("int16")
            self.mean_buffer_counter = 1
            # else:
            #     self.mean_buffer_counter += 1
        else:
            self.mean_buffer_counter += 1

        return self.current_mean.astype("int16") - self.long_mean

    def calc_difference_from_buffer(self):
        self.long_mean = np.mean(self.framebuffer, axis=2).astype("int16")
        self.current_mean = np.mean(
            self.framebuffer[:, :, : self.current_mean_frames], axis=2
        ).astype("uint8")
        return self.current_mean.astype("int16") - self.long_mean

    def threshold_diff(
        self,
        diff,
        threshold=2,
    ):
        self.long_std_dev = np.std(self.framebuffer, axis=2).astype("uint8")
        diff[abs(diff) < threshold * self.long_std_dev] = 0
        return diff

    def mask_regions(self, img, area="sonar_controls"):
        if area == "non_object_space":
            if img.shape[:1] != self.non_object_space_mask.shape[:1]:
                percent_difference = (
                    img.shape[0] / self.non_object_space_mask.shape[0] * 100
                )

                np.place(
                    img,
                    FishDetector.resize_img(
                        self.non_object_space_mask, percent_difference
                    )
                    < 100,
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
                    FishDetector.resize_img(
                        self.sonar_controls_mask, percent_difference
                    )
                    < 100,
                    0,
                )
            else:
                np.place(img, self.sonar_controls_mask < 100, 0)
        return img

    @staticmethod
    def rgb_to_gray(img):
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    @staticmethod
    def spatial_filter(img, kernel_size=10, method="average"):
        if method == "average":
            return cv.blur(img, (kernel_size, kernel_size))
        if method == "median":
            return cv.medianBlur(img, kernel_size)

    def get_new_id(self):
        if self.latest_obj_index > 300000:
            self.latest_obj_index = 0
        self.latest_obj_index += 1
        return self.latest_obj_index

    def is_duplicate(self, img, threshold=25):
        if self.framebuffer is None:
            return False
        elif (
            np.mean(abs(img - self.framebuffer[:, :, self.framebuffer.shape[2]]))
            < threshold
        ):
            print("Duplicate frame.")
            return True

    @staticmethod
    def resize_img(img, scale_percent):
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        return cv.resize(img, dim, interpolation=cv.INTER_AREA)
