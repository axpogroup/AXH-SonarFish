import cv2 as cv
import numpy as np
from Object import Object

fish_area_mask = cv.imread("masks/fish.png", cv.IMREAD_GRAYSCALE)
full_area_mask = cv.imread("masks/full.png", cv.IMREAD_GRAYSCALE)


class FishDetector:
    def __init__(self, filename):
        self.filename = filename
        self.frame_number = 0  # TOD Must not overflow - recycle
        self.total_runtime_ms = None

        self.current_raw = None
        self.current_gray = None
        self.current_enhanced = None
        self.current_output = None
        self.current_classified = None

        # Enhancement
        self.downsample = None
        self.framebuffer = None
        self.mean_buffer = None
        self.mean_buffer_counter = None
        self.buffer_size = 120
        self.long_mean_frames = 120
        self.long_mean = None
        self.long_std_dev = None
        self.current_mean_frames = 10
        self.current_mean = None
        self.enhance_time_ms = None
        self.current_long_mean_uint8 = None

        # Detection and Tracking
        self.detections = {}
        self.current_objects = {}
        self.current_threshold = None
        self.associations = []
        self.max_obj_index = 0
        self.max_association_dist = 60
        self.phase_out_after_x_frames = 5
        self.min_occurences_in_last_x_frames = (13, 15)
        self.detection_tracking_time_ms = None

        # Classification
        self.river_pixel_velocity = (-1.91, -0.85)
        self.rotation_rad = -2.7229

    def process_frame(self, raw_frame, secondary=None, downsample=False):
        start = cv.getTickCount()
        if downsample:
            self.downsample = 25
            raw_frame = self.resize_img(raw_frame, 25)
        self.current_raw = raw_frame
        self.current_gray = self.rgb_to_gray(self.current_raw)
        self.current_enhanced = self.enhance_frame(self.current_gray)
        self.enhance_time_ms = int(
            (cv.getTickCount() - start) / cv.getTickFrequency() * 1000
        )

        # if self.mean_buffer.shape[2] == int(self.long_mean_frames / self.current_mean_frames):
        if self.frame_number > self.buffer_size:
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
        light = False
        enhanced_temp = self.mask_regions(gray_frame, area="fish")

        if light:
            self.update_buffer_light(enhanced_temp)
            if self.frame_number < self.buffer_size:
                return enhanced_temp * 0
        else:
            self.update_buffer(enhanced_temp)
            if self.frame_number < self.buffer_size:
                return enhanced_temp * 0

        if light:
            enhanced_temp = self.calc_difference_from_buffer_light()
            enhanced_temp[abs(enhanced_temp) < 20] = 0
        else:
            enhanced_temp = self.calc_difference_from_buffer()
            enhanced_temp = self.threshold_diff(enhanced_temp, threshold=2)

        self.current_long_mean_uint8 = self.long_mean.astype("uint8")

        # # Transform into visual/uint8 image
        enhanced_temp = (abs(enhanced_temp) + 125).astype("uint8")
        # ret, self.current_enhanced = cv.threshold(self.current_enhanced, 160, 255, 0)
        enhanced_temp = self.spatial_filter(
            enhanced_temp, kernel_size=3, method="median"
        )
        return enhanced_temp

    def detect_and_track(self, enhanced_frame):
        self.detections = self.find_points_of_interest(enhanced_frame, mode="contour")
        self.current_objects = self.associate_detections(self.detections)
        self.current_objects = self.filter_objects(self.current_objects)
        return

    def draw_output(self, img, classifications=False, debug=False, runtiming=False, fullres=False):
        output = self.retrieve_frame(img)
        output = self.draw_objects(output, classifications=classifications, debug=debug, fullres=fullres)
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
                f"{self.total_runtime_ms} ms - Total - FPS: {int(1000/self.total_runtime_ms)}",
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

    def draw_objects(self, img, debug=False, classifications=False, fullres=False):
        for ID, obj in self.current_objects.items():
            if obj.show[-1]:
                if classifications:
                    if fullres:
                        obj.draw_classifications_box(img, self.downsample)
                    else:
                        obj.draw_classifications_box(img)
                else:
                    if obj.classification[-1] == "Fisch":
                        obj.draw_bounding_box(img, color=(0, 255, 0))
                        obj.draw_past_midpoints(img, color=(0, 255, 0))
                    else:
                        obj.draw_bounding_box(img, color=(255, 0, 0))
                        obj.draw_past_midpoints(img, color=(255, 0, 0))
            elif (obj.frames_observed[-1] == self.frame_number) & debug:
                obj.draw_bounding_box(img, color=(20, 20, 20))
                obj.draw_past_midpoints(img, color=(20, 20, 20))

            elif debug:
                obj.draw_bounding_box(img, color=(50, 50, 50))

            # if debug:
            # cv.circle(img, (obj.midpoints[-1][0], obj.midpoints[-1][1]),
            #           int(self.max_association_dist/2), (0, 0, 255), 1)

        return img

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
                enhanced_frame.astype("uint8"), (25, 25), 0
            )

            # Threshold
            ret, thres = cv.threshold(enhanced_frame, 127, 255, 0)
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
                new_object = Object(self.get_new_id(), contour, self.frame_number)
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

        if self.framebuffer.shape[2] > self.buffer_size:
            self.framebuffer = self.framebuffer[:, :, : self.buffer_size]

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
        self.current_mean = np.mean(self.framebuffer[:, :, :10], axis=2).astype("uint8")
        return self.current_mean.astype("int16") - self.long_mean

    def threshold_diff(
        self,
        diff,
        threshold=2,
    ):
        self.long_std_dev = np.std(self.framebuffer, axis=2).astype("uint8")
        diff[abs(diff) < threshold * self.long_std_dev] = 0
        return diff

    def create_mean_std_dev(
        self,
    ):  # Unused for now since the ground pattern changes we can't use a hardcoded image
        # Put in process_frame()
        # Do this to save time before implementing it in a rolling manner
        # self.create_mean_std_dev()
        # quit()
        # self.current_mean = np.mean(self.framebuffer, axis=2).astype('uint8')

        # Put in initialization
        # fname_temp = os.path.split(self.filename)
        # self.mean_stddev_file = (
        #         fname_temp[0] + "/mean_std_dev/" + fname_temp[1] + "_mean_stddev_.npz"
        # )
        # try:
        #     temp = np.load(self.mean_stddev_file)
        #     self.long_mean = temp["mean"]
        #     self.long_std_dev = temp["std_dev"]
        # except FileNotFoundError:
        #     print("No existing mean and std-dev file")
        #     self.long_mean = None
        #     self.long_std_dev = None

        long_mean = np.mean(self.framebuffer, axis=2)
        long_std_dev = np.std(self.framebuffer, axis=2)
        np.savez(self.mean_stddev_file, mean=long_mean, std_dev=long_std_dev)

    @staticmethod
    def mask_regions(img, area="fish"):
        if area == "fish":
            if img.shape[:1] != fish_area_mask.shape[:1]:
                percent_difference = img.shape[0] / fish_area_mask.shape[0] * 100

                np.place(img, FishDetector.resize_img(fish_area_mask, percent_difference) < 100, 0)
            else:
                np.place(img, fish_area_mask < 100, 0)
        elif area == "full":
            if img.shape[:1] != full_area_mask.shape[:1]:
                percent_difference = img.shape[0] / full_area_mask.shape[0] * 100

                np.place(img, FishDetector.resize_img(full_area_mask, percent_difference) < 100, 0)
            else:
                np.place(img, full_area_mask < 100, 0)
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

    @staticmethod
    def retrieve_frame(img):
        if img is None:
            return np.zeros((270, 480, 3), dtype=np.uint8)
        elif len(img.shape) == 3:
            if img.shape[2] == 3:
                return img

        return cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    def get_new_id(self):
        if self.max_obj_index > 300000:
            self.max_obj_index = 0
        self.max_obj_index += 1
        return self.max_obj_index

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