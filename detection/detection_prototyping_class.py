import os
from math import cos, sin

import cv2 as cv
import numpy as np

fish_area_mask = cv.imread("masks/fish.png", cv.IMREAD_GRAYSCALE)
full_area_mask = cv.imread("masks/full.png", cv.IMREAD_GRAYSCALE)


class Object:
    def __init__(self, identifier, contour, frame_number):
        self.ID = identifier
        self.frames_observed = [frame_number]
        self.show = [False]
        self.contours = [contour]

        x, y, w, h = cv.boundingRect(contour)
        self.midpoints = [(int(x + w / 2), int(y + h / 2))]

        self.classification = ["Objekt"]

        self.velocity = []
        self.mean_v = None
        self.mean_v_last_10 = None
        self.river_rot = -2.7229

    def update_object(self, detection):
        self.frames_observed.append(detection.frames_observed[-1])
        self.show.append(False)
        self.contours.append(detection.contours[-1])
        self.midpoints.append(detection.midpoints[-1])

        self.calculate_speed()
        self.classify_object()

    def draw_bounding_box(self, img, color):
        x, y, w, h = cv.boundingRect(self.contours[-1])
        cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv.putText(
            img,
            (self.classification[-1] + " " + str(self.ID)),
            (x, y - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
        )
        # if self.mean_v is not None:
        #     cv.putText(img, (str(self.mean_v[0])),
        #                      (x, y + 10), cv.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        return img

    def draw_classifications_box(self, img):
        x, y, w, h = cv.boundingRect(self.contours[-1])
        if self.classification[-1] == "Fisch":
            color = (0, 255, 0)
        else:
            color = (250, 150, 150)

        cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv.putText(
            img,
            (self.classification[-1]),
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

    def classify_object(self):
        fish = False
        if len(self.velocity) < 20:
            fish = False
            return

        if self.classification[-1] == "Fisch":
            if abs(self.mean_v[1]) > 0.5:
                fish = True
            elif abs(self.mean_v[0] - 2.185) > 1:
                fish = True

        if abs(self.mean_v_last_10[1]) > 1:
            fish = True
        elif abs(self.mean_v_last_10[0] - 2.185) > 2:
            fish = True

        if fish:
            self.classification.append("Fisch")
        else:
            self.classification.append("Objekt")
        return

    def rotate_vector(self, vec, theta):
        rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
        return np.dot(rot, vec)


class FishDetector:
    def __init__(self, filename):
        self.filename = filename
        self.current_frame = None
        self.framebuffer = None
        self.buffer_size = 120
        self.frame_number = 0  # TOD Must not overflow - recycle

        self.current_mean = None
        self.current_enhanced = None
        self.current_tracked = None
        self.current_labeled = None
        self.current_output = None
        self.points_of_interest_filter = None
        self.raw_frame = None

        # Detection and Tracking
        self.detections = {}
        self.current_objects = {}
        self.associations = []
        self.max_obj_index = 0
        self.max_association_dist = 60

        self.phase_out_after_x_frames = 5
        self.min_occurences_in_last_x_frames = (13, 15)

        # Classification
        self.river_pixel_velocity = (-1.91, -0.85)
        self.rotation_rad = -2.7229

        fname_temp = os.path.split(self.filename)
        self.mean_stddev_file = (
            fname_temp[0] + "/mean_std_dev/" + fname_temp[1] + "_mean_stddev_.npz"
        )
        try:
            temp = np.load(self.mean_stddev_file)
            self.long_mean = temp["mean"]
            self.long_std_dev = temp["std_dev"]
        except FileNotFoundError:
            print("No existing mean and std-dev file")
            self.long_mean = None
            self.long_std_dev = None

    def process_frame(self, frame, raw_frame):
        self.raw_frame = raw_frame
        self.current_frame = self.rgb_to_gray(frame)
        # self.current_output = self.enhance_frame(frame)
        self.current_output = self.detect_and_track(self.current_frame)

        # self.current_output = self.mask_regions(self.current_output, area='fish')
        self.frame_number += 1
        return self.current_output

    def enhance_frame(self):
        self.current_frame = self.mask_regions(self.current_frame, area="fish")
        self.update_buffer(self.current_frame)

        if self.framebuffer.shape[2] < self.buffer_size:
            return self.current_frame * 0

        self.current_enhanced = self.calc_difference()
        # self.current_enhanced = self.threshold_diff(self.current_enhanced, threshold=2)

        # # Transform into visual image
        self.current_enhanced = (abs(self.current_enhanced) + 125).astype("uint8")
        # ret, self.current_enhanced = cv.threshold(self.current_enhanced, 160, 255, 0)
        self.current_enhanced = self.spatial_filter(
            self.current_enhanced, kernel_size=15, method="median"
        )
        return self.current_enhanced

    def detect_and_track(self, frame):
        self.find_points_of_interest(frame, mode="contour")
        self.associate_detections()
        self.current_output = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

        self.filter_objects()

        # for detection_id, detection in self.detections.items():
        #     self.current_output = detection.draw_bounding_box(self.current_output, (255, 0, 0))
        # for current_object_id, current_object in self.current_objects.items():
        #     self.current_output = current_object.draw_bounding_box(self.current_output, (0, 255, 0))

        # self.current_output = self.draw_associations(self.current_output, color=(200, 230, 0))
        raw_output = self.draw_objects(self.raw_frame, classifications=True)
        self.current_output = self.draw_objects(self.current_output, debug=True)
        # self.current_tracked = self.draw_objects(cv.cvtColor(frame, cv.COLOR_GRAY2BGR), debug=True)

        return self.current_output, raw_output

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

    def draw_objects(self, img, debug=False, classifications=False):
        for ID, obj in self.current_objects.items():
            if obj.show[-1]:
                if classifications:
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

    def filter_objects(self):
        to_delete = []
        for ID, obj in self.current_objects.items():
            # Delete if it hasn't been observed in the last 5 frames
            if (
                self.frame_number - obj.frames_observed[-1]
                > self.phase_out_after_x_frames
            ):
                to_delete.append(ID)
                continue

            # Show if 8 occurences in the last 10 frames
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
            self.current_objects.pop(key)

    def associate_detections(self):
        if len(self.current_objects) == 0:
            self.current_objects = self.detections
            return

        object_midpoints = [
            existing_object.midpoints[-1]
            for _, existing_object in self.current_objects.items()
        ]
        object_ids = list(self.current_objects.keys())
        new_objects = []
        self.associations = []
        for detection_id, detection in self.detections.items():
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
                self.detections[association["detection_id"]]
            )

        return

    @staticmethod
    def closest_point(point, points):
        points = np.asarray(points)
        dist_2 = np.sqrt(np.sum((points - point) ** 2, axis=1))
        min_index = np.argmin(dist_2)
        return min_index, dist_2[min_index]

    def find_points_of_interest(self, img, mode="contour"):
        if mode == "contour":
            # Make positive and negative differences the same
            img = (abs(img.astype("int16") - 125) + 125).astype("uint8")

            # Consolidate the points
            img = cv.GaussianBlur(img.astype("uint8"), (101, 101), 0)

            # Threshold
            ret, thres = cv.threshold(img, 127, 255, 0)

            # Alternative consolidation - dilate
            # kernel = np.ones((51, 51), 'uint8')
            # thres = cv.dilate(thres, kernel, iterations=1)
            # img = self.spatial_filter(img, kernel_size=15, method='median')

            contours, hier = cv.findContours(
                thres, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )
            self.points_of_interest_filter = thres
            self.detections = {}
            for contour in contours:
                new_object = Object(self.get_new_id(), contour, self.frame_number)
                self.detections[new_object.ID] = new_object

            return self.detections

        elif mode == "blob":  # TOD there is an issue, it Segmentation faults instantly
            blob_detector = cv.SimpleBlobDetector()
            keypoints = blob_detector.detect(img)
            # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
            im_with_keypoints = cv.drawKeypoints(
                img,
                keypoints,
                img,
                (0, 0, 255),
                cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )
            return im_with_keypoints

    def update_buffer(self, img):
        if self.framebuffer is None:
            self.framebuffer = img[:, :, np.newaxis]
        else:
            self.framebuffer = np.concatenate(
                (img[..., np.newaxis], self.framebuffer), axis=2
            )

        if self.framebuffer.shape[2] > self.buffer_size:
            self.framebuffer = self.framebuffer[:, :, : self.buffer_size]

    def calc_difference(self):
        if self.long_mean is None:
            print("WARNING: No long mean provided, calculating from buffer.")
            self.long_mean = np.mean(self.framebuffer, axis=2).astype("int16")

        self.current_mean = np.mean(self.framebuffer[:, :, :10], axis=2).astype("uint8")
        return self.current_mean.astype("int16") - self.long_mean

    def threshold_diff(self, diff, threshold=2):
        if self.long_std_dev is None:
            print("WARNING: No std dev provided, calculating from buffer.")
            self.long_std_dev = np.std(self.framebuffer, axis=2).astype("uint8")

        diff[abs(diff) < threshold * self.long_std_dev] = 0
        return diff

    def create_mean_std_dev(self):
        # Put in process_frame()
        # Do this to save time before implementing it in a rolling manner
        # self.create_mean_std_dev()
        # quit()
        # self.current_mean = np.mean(self.framebuffer, axis=2).astype('uint8')

        long_mean = np.mean(self.framebuffer, axis=2)
        long_std_dev = np.std(self.framebuffer, axis=2)
        np.savez(self.mean_stddev_file, mean=long_mean, std_dev=long_std_dev)

    @staticmethod
    def mask_regions(img, area="fish"):
        if area == "fish":
            np.place(img, fish_area_mask < 100, 0)
        elif area == "full":
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


if __name__ == "__main__":
    enhanced = True
    # recording_file = "recordings/Schwarm_einzel_jet_to_gray_snippet.mp4"
    recording_file = "output/normed_120_10_std_dev_threshold_2_median_11_drop_duplicates_crop.mp4"  # enhanced
    # recording_file = "output/components/final_old_moving_average_5s.mp4"  # enhanced
    # recording_file = "recordings/new_settings/22-11-14_start_15-21-23_crop.mp4"

    write_file = False
    # output_file = "output/components/normed_120_10_std_dev_threshold_2_median_11_schwarm_temp.mp4"
    output_file = "output/components/presentation_v1.mp4"
    # output_file = "output/normed_120_minus_10.mp4"
    # output_file = "output/normed_120_10_std_dev_threshold_2.mp4"

    # Input
    video_cap = cv.VideoCapture(recording_file)
    frame_by_frame = False
    previous_img = False

    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(video_cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv.CAP_PROP_FRAME_HEIGHT)) * 2
    fps = int(video_cap.get(cv.CAP_PROP_FPS))

    # initialize the FourCC and a video writer object
    fourcc = cv.VideoWriter_fourcc("m", "p", "4", "v")
    video_writer = cv.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    # Initialize FishDetector Instance
    detector = FishDetector(recording_file)

    frame_no = 0
    frames_total = int(video_cap.get(cv.CAP_PROP_FRAME_COUNT))
    while video_cap.isOpened():
        ret, raw_frame = video_cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Start timer
        frame_no += 1
        timer = cv.getTickCount()

        # Detection
        if enhanced:
            enhanced_frame = raw_frame[:1080, :, :]
            output, raw_classified = detector.process_frame(
                enhanced_frame, raw_frame[1080:, :, :]
            )
        else:
            output = detector.process_frame(raw_frame)

        # Calculate Frames per second (FPS)
        fps = cv.getTickFrequency() / (cv.getTickCount() - timer)

        # Output
        if enhanced:
            disp = np.concatenate((output, raw_classified))
        else:
            disp = np.concatenate((output, raw_frame))
        cv.putText(
            disp,
            "FPS : " + str(int(fps)),
            (150, 50),
            cv.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )
        cv.imshow("frame", disp)
        if write_file:
            # disp = cv.cvtColor(disp, cv.COLOR_GRAY2BGR)
            video_writer.write(disp)
        if frame_no % 20 == 0:
            print(f"Processed {frame_no/frames_total*100} % of video.")
            if frame_no / frames_total * 100 > 35:
                pass

        if not frame_by_frame:
            usr_input = cv.waitKey(1)
        if usr_input == ord(" "):
            if cv.waitKey(0) == ord(" "):
                frame_by_frame = True
            else:
                frame_by_frame = False
            print("Press any key to continue ... ")
        if usr_input == 27:
            break

    video_cap.release()
    video_writer.release()
    cv.destroyAllWindows()
