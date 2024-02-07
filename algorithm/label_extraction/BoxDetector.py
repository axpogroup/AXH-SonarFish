# Code written by Leiv Andresen, HTD-A, leiv.andresen@axpo.com

import copy

import cv2 as cv
import numpy as np

from .BoxObject import BoxObject
from utils import resize_img

fish_area_mask = None
full_area_mask = None


class BoxDetector:
    def __init__(self, settings_dict, latest_persistent_object_id):
        self.settings_dict = settings_dict
        self.frame_number = 0  # TOD Must not overflow - recycle
        self.total_runtime_ms = None
        self.current_raw = None
        self.current_gray = None
        self.current_enhanced = None
        self.current_output = None
        self.current_classified = None

        # colorchannels
        self.current_red = None
        self.current_green = None
        self.current_blue = None

        # Issue occured
        self.issue = False

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
        self.latest_persistent_object_id = latest_persistent_object_id
        self.detection_tracking_time_ms = None
        self.max_association_dist = settings_dict["max_association_dist"]
        self.phase_out_after_x_frames = settings_dict["phase_out_after_x_frames"]
        self.min_occurences_in_last_x_frames = settings_dict[
            "min_occurences_in_last_x_frames"
        ]

        # Classification
        self.river_pixel_velocity = settings_dict["river_pixel_velocity"]
        self.rotation_rad = settings_dict["rotation_rad"]

    def extract_green_red(self, current_raw):
        green = copy.deepcopy(current_raw[:, :, 1])
        np.place(  # blue higher 150, more red than green
            green,
            (
                (current_raw[:, :, 1] < 200)
                | (current_raw[:, :, 0] > 150)
                | (current_raw[:, :, 2] > 150)
            ),
            0,
        )

        red = copy.deepcopy(current_raw[:, :, 2])
        # np.place(
        #     red, current_raw[:, :, 0] > 150,
        #     0,
        # )
        np.place(  # blue higher 150, more red than green
            red,
            ((current_raw[:, :, 0] > 150) | (current_raw[:, :, 1] > 150)),
            0,
        )
        return green, red

    def extract_green_red_blue_no_background(self, current_raw):
        current_raw = cv.GaussianBlur(
            current_raw,
            (self.blur_filter_kernel, self.blur_filter_kernel),
            0,
        )
        green = copy.deepcopy(current_raw[:, :, 1])

        red = copy.deepcopy(current_raw[:, :, 2])

        blue = copy.deepcopy(current_raw[:, :, 0])
        return green, red, blue

    def process_frame(self, raw_frame):
        self.issue = False
        start = cv.getTickCount()

        self.current_raw = resize_img(raw_frame, 25)
        (
            self.current_green,
            self.current_red,
            self.current_blue,
        ) = self.extract_green_red_blue_no_background(self.current_raw)

        self.enhance_time_ms = int(
            (cv.getTickCount() - start) / cv.getTickFrequency() * 1000
        )

        self.detect_and_track(self.current_blue, color="blue")
        self.detect_and_track(self.current_green, color="green")
        self.detect_and_track(self.current_red, color="red")

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
        # enhanced_temp = self.mask_regions(gray_frame, area="fish")
        enhanced_temp = gray_frame
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
            enhanced_temp[abs(enhanced_temp) < 20] = 0
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

    def detect_and_track(self, enhanced_frame, color=None):
        self.detections = self.find_points_of_interest(enhanced_frame, mode="contour")

        for key, detection in self.detections.items():
            detection.classifications = [color]

        self.current_objects = self.associate_detections(self.detections, color=color)

        self.current_objects = self.filter_objects(self.current_objects, color=color)
        return

    def draw_output(
        self,
        img,
        debug=False,
        runtiming=False,
        only_runtime=False,
    ):
        output = self.retrieve_frame(img)
        if not only_runtime:
            output = self.draw_objects(output, debug=debug)
        if runtiming:
            cv.rectangle(output, (1390, 25), (1850, 155), (0, 0, 0), -1)
            color = (255, 255, 255)
            cv.putText(
                output,
                f"Frame no. {self.frame_number}",
                (1500, 50),
                cv.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
            )
            cv.putText(
                output,
                f"{self.enhance_time_ms} ms - Enhancement",
                (1400, 80),
                cv.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
            )
            cv.putText(
                output,
                f"{self.detection_tracking_time_ms} ms - Detection & Tracking",
                (1400, 110),
                cv.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
            )
            if self.total_runtime_ms > 40:
                color = (100, 100, 255)
            if self.total_runtime_ms == 0:
                self.total_runtime_ms = 1
            cv.putText(
                output,
                f"{self.total_runtime_ms} ms - Total - FPS: {int(1000 / self.total_runtime_ms)}",
                (1400, 140),
                cv.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
            )
        return output

    def draw_associations(self, img, color):
        for association in self.associations:
            cv.line(
                img,
                self.detections[association["detection_id"]].midpoints[-1],
                self.current_objects[association["existing_object_id"]].midpoints[-1],
                color,
                2,
            )
            cv.putText(
                img,
                str(association["distance"]),
                self.detections[association["detection_id"]].midpoints[-1],
                cv.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
            )
        return img

    def draw_objects(self, img, debug=False):
        for ID, obj in self.current_objects.items():

            if debug:
                obj.draw_bounding_box(img, color=(200, 200, 200))
                obj.draw_past_midpoints(img, color=(200, 200, 200))

        return img

    def filter_objects(self, current_objects, color=None):
        to_delete = []
        for ID, obj in current_objects.items():
            if color is not None and obj.classifications[-1] != color:
                continue

            # Delete if it hasn't been observed in the last x frames
            if (
                self.frame_number - obj.frames_observed[-1]
                >= self.phase_out_after_x_frames
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

    def associate_detections(self, detections, color=None):
        if len(self.current_objects) == 0:
            if len(detections) == 0:
                self.current_objects = detections
                return self.current_objects
            else:
                for detection_id, detection in detections.items():
                    detection.persistent_id = self.latest_persistent_object_id
                    self.latest_persistent_object_id += 1
                    self.current_objects[detection.ID] = detection
                    return self.current_objects

        object_midpoints = [
            existing_object.midpoints[-1]
            for _, existing_object in self.current_objects.items()
            if existing_object.classifications[-1] == color
        ]
        object_ids = [
            ID
            for ID, existing_object in self.current_objects.items()
            if existing_object.classifications[-1] == color
        ]

        new_objects = []
        self.associations = []
        if len(object_ids) == 0:
            for detection_id, detection in detections.items():
                if detection.classifications[-1] == color:
                    new_objects.append(detection)

        else:
            for detection_id, detection in detections.items():
                min_id, min_dist = self.closest_point(
                    detection.midpoints[-1], object_midpoints
                )
                max_association_dist = self.max_association_dist
                if min_dist < max_association_dist:
                    self.associations.append(
                        {
                            "detection_id": detection.ID,
                            "existing_object_id": object_ids[min_id],
                            "distance": min_dist,
                        }
                    )
                else:
                    new_objects.append(detection)

        for association in self.associations:
            self.current_objects[association["existing_object_id"]].update_object(
                detections[association["detection_id"]]
            )

        for new_object in new_objects:
            new_object.persistent_id = self.latest_persistent_object_id
            self.latest_persistent_object_id += 1
            self.current_objects[new_object.ID] = new_object

        return self.current_objects

    @staticmethod
    def closest_point(point, points):
        points = np.asarray(points)
        dist_2 = np.sqrt(np.sum((points - point) ** 2, axis=1))
        min_index = np.argmin(dist_2)
        return min_index, dist_2[min_index]

    def find_points_of_interest(self, enhanced_frame, mode="contour"):
        if mode == "contour":
            ret, thres = cv.threshold(enhanced_frame, self.threshold_contours, 255, 0)

            self.current_threshold = thres

            contours, hier = cv.findContours(
                thres, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
            )
            detections = {}
            for contour in contours:
                new_object = BoxObject(self.get_new_id(), contour, self.frame_number)
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

    @staticmethod
    def rgb_to_gray(img):
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    @staticmethod
    def spatial_filter(img, kernel_size=10, method="average"):
        if method == "average":
            return cv.blur(img, (kernel_size, kernel_size))
        if method == "median":
            return cv.medianBlur(img, kernel_size)

    @staticmethod
    def retrieve_frame(img, puttext=None):
        out = copy.deepcopy(img)
        if out is None:
            out = np.zeros((270, 480, 3), dtype=np.uint8)
            if puttext is not None:
                cv.putText(
                    out,
                    puttext,
                    (50, 50),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                )
            return out

        elif len(out.shape) == 3:
            if out.shape[2] == 3:
                if puttext is not None:
                    cv.putText(
                        out,
                        puttext,
                        (50, 50),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (255, 255, 255),
                        2,
                    )
                return out

        if puttext is not None:
            cv.putText(
                out,
                puttext,
                (50, 50),
                cv.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )

        return cv.cvtColor(out, cv.COLOR_GRAY2BGR)

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

    def prepare_objects_for_csv(self, timestr, file):
        rows = []
        if self.current_objects is not None:
            for _, object_ in self.current_objects.items():
                # area = cv.contourArea(object_.contours[-1])
                x, y, w, h = cv.boundingRect(object_.contours[-1])
                row = [
                    f"{self.frame_number}",
                    f"{object_.persistent_id}",
                    f"{x}",
                    f"{y}",
                    str(w),
                    str(h),
                    "-1",
                    "-1",
                    "-1",
                    "-1",
                ]
                rows.append(row)
        return rows
