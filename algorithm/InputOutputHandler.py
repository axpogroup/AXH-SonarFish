import csv
import datetime as dt
import os

import cv2 as cv
import pandas as pd

import algorithm.visualization_functions as visualization_functions
from algorithm.utils import get_elapsed_ms


class InputOutputHandler:
    def __init__(self, settings_dict):
        self.csv_file = None
        self.csv_writer = None
        self.video_writer = None
        self.settings_dict = settings_dict
        self.video_cap = cv.VideoCapture(self.settings_dict["input_file"])
        self.input_filename = os.path.split(self.settings_dict["input_file"])[-1]
        self.output_dir_name = None

        if (
            "record_output_video" in settings_dict.keys()
            and self.settings_dict["record_output_video"]
        ):
            self.initialize_output_recording()

        if (
            "record_output_csv" in settings_dict.keys()
            and self.settings_dict["record_output_csv"]
        ):
            self.initialize_output_csv()

        try:
            date_fmt = "%y-%m-%d_start_%H-%M-%S.mp4"
            self.start_datetime = dt.datetime.strptime(
                os.path.split(settings_dict["input_file"])[-1], date_fmt
            )
        except ValueError:
            self.start_datetime = dt.datetime(year=2000, month=1, day=1)

        if "display_trackbars" in settings_dict.keys():
            self.display_trackbars = settings_dict["display_trackbars"]
        else:
            self.display_trackbars = False
        self.frame_by_frame = False
        self.usr_input = None
        self.frame_no = 0
        self.frames_total = int(self.video_cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.video_cap.get(cv.CAP_PROP_FPS))
        self.start_ticks = 1

        self.frame_retrieval_time = None
        self.last_output_time = None

        self.current_raw_frame = None

    def get_new_frame(self):
        start = cv.getTickCount()
        tries = 0
        if self.video_cap.isOpened():
            while tries < 5:
                ret, self.current_raw_frame = self.video_cap.read()
                self.frame_retrieval_time = get_elapsed_ms(start)

                # if frame is read correctly ret is True
                if not ret:
                    tries += 1
                else:
                    self.frame_no += 1
                    return True

            print("Can't receive frame (stream end?). Exiting ...")
            self.shutdown()
            return False

        else:
            print("ERROR: Video Capturer is not open.")
            return False

    @staticmethod
    def get_detections_pd(object_history):
        rows = []
        for _, obj in object_history.items():
            for i in range(len(obj.frames_observed)):
                rows.append(
                    [
                        obj.ID,
                        obj.frames_observed[i],
                        obj.midpoints[i][0],
                        obj.midpoints[i][1],
                        obj.bounding_boxes[i][0],
                        obj.bounding_boxes[i][1],
                        obj.velocities[i][0],
                        obj.velocities[i][1],
                        obj.velocities_rot[i][0],
                        obj.velocities_rot[i][1],
                        obj.areas[i],
                    ]
                )

        return pd.DataFrame(
            rows, columns=["ID", "frames_observed", "x", "y", "width", "height", "v_x", "v_y", "vrotcode_x", "vrotcode_y", "contour_area"]
        )

    def trackbars(self, detector):
        # settings = ["short_mean_frames", "long_mean_frames"]
        #
        # def change(value):
        #     for setting in settings:
        #         val = cv.getTrackbarPos(setting, 'frame')
        #         if (setting == "short_mean_frames") and val == 0:
        #             val = 1
        #         if (setting == "long_mean_frames") and val < detector.conf["short_mean_frames"]:
        #             val = detector.conf["short_mean_frames"]
        #         if (setting in ["dilation_kernel_m", "median_filter_kernel_m"]) and (float(val) / 2 % 1) == 0:
        #             val = int(val) + 1
        #
        #         detector.conf[setting] = val
        #
        # # TOD0 for some reason it doesn't work if the setting is shown in text - weird
        # for setting in settings:
        #     name = str(detector.conf[setting])  + "_" + setting
        #     cv.createTrackbar(
        #         name, "frame", detector.conf[setting], int(detector.conf[setting]*2), change
        #     )

        def change_current_mean_frames(value):
            if value == 0:
                value = 1
            detector.conf["short_mean_frames"] = value

        def change_long_mean_frames(value):
            if value < detector.current_mean_frames:
                detector.conf["long_mean_frames"] = detector.conf["short_mean_frames"]
            else:
                detector.conf["long_mean_frames"] = value

        def change_alpha(value):
            detector.conf["contrast"] = float(value) / 10

        def change_beta(value):
            detector.conf["brightness"] = value

        def change_diff_thresh(value):
            detector.conf["difference_threshold_scaler"] = value / 10

        def change_median_filter_kernel(value):
            detector.conf["median_filter_kernel_mm"] = value

        def change_dilatation_kernel(value):
            detector.conf["dilation_kernel_mm"] = value

        cv.createTrackbar(
            "contrast*10",
            "frame",
            int(detector.conf["contrast"] * 10),
            30,
            change_alpha,
        )
        cv.createTrackbar(
            "brightness", "frame", detector.conf["brightness"], 120, change_beta
        )
        cv.createTrackbar(
            "s_mean",
            "frame",
            detector.conf["short_mean_frames"],
            120,
            change_current_mean_frames,
        )
        cv.createTrackbar(
            "l_mean",
            "frame",
            detector.conf["long_mean_frames"],
            1200,
            change_long_mean_frames,
        )
        cv.createTrackbar(
            "diff_thresh*10",
            "frame",
            int(detector.conf["difference_threshold_scaler"] * 10),
            127,
            change_diff_thresh,
        )
        cv.createTrackbar(
            "median_f",
            "frame",
            detector.conf["median_filter_kernel_mm"],
            1200,
            change_median_filter_kernel,
        )
        cv.createTrackbar(
            "dilate",
            "frame",
            detector.conf["dilation_kernel_mm"],
            1200,
            change_dilatation_kernel,
        )

    def show_image(self, img, detector):
        cv.imshow("frame", img)
        if self.display_trackbars:
            self.trackbars(detector)

        if not self.frame_by_frame:
            self.usr_input = cv.waitKey(1)

        # If pause button was pressed, then the pause button can be used to playback frame by frame.
        # Any other button will resume playback
        if self.usr_input == ord(" "):
            print("Press any key to continue playback or SPACE for next frame ... ")
            if cv.waitKey(0) == ord(" "):
                self.frame_by_frame = True
            else:
                self.frame_by_frame = False
            return

        if self.usr_input == 27:
            print("User pressed ESC, closing video stream ...")
            self.shutdown()
            return

    def handle_output(self, processed_frame, object_history, runtimes, detector):
        # Total runtime
        if self.last_output_time is not None:
            total_time_per_frame = get_elapsed_ms(self.last_output_time)
        else:
            self.start_ticks = cv.getTickCount()
        total_runtime = get_elapsed_ms(self.start_ticks)
        self.last_output_time = cv.getTickCount()

        if self.frame_no % 20 == 0:
            if total_time_per_frame == 0:
                total_time_per_frame = 1
            print(
                f"Processed {'{:.1f}'.format(self.frame_no / self.frames_total * 100)} % of video. "
                f"Runtimes [ms]: getFrame: {self.frame_retrieval_time} | Enhance: {runtimes['enhance']} | "
                f"DetectTrack: {runtimes['detection_tracking']} | "
                f"Total: {total_time_per_frame} | FPS: {'{:.1f}'.format(self.frame_no/(2*total_runtime/1000))}"
            )

        if "record_output_csv" in self.settings_dict.keys():
            self.write_csv(detector)

        if ("display_output_video" in self.settings_dict.keys()) or (
            "record_output_video" in self.settings_dict.keys()
        ):
            extensive = (
                False
                if "display_mode_extensive" not in self.settings_dict.keys()
                else self.settings_dict["display_mode_extensive"]
            )
            disp = visualization_functions.get_visual_output(
                detector, processed_frame, extensive=extensive
            )

            if "record_output_video" in self.settings_dict.keys():
                self.video_writer.write(disp)

            if (
                "display_output_video" in self.settings_dict.keys()
                and self.settings_dict["display_output_video"]
            ):
                self.show_image(disp, detector)

    def initialize_output_recording(self):
        # grab the width, height, fps and length of the video stream.
        frame_width = int(self.video_cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.video_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.video_cap.get(cv.CAP_PROP_FPS))

        self.output_dir_name = os.path.join(
            self.settings_dict["output_directory"],
            dt.datetime.now().strftime("%y_%m_%d_%H-%M-%S_")
            + self.settings_dict["tag"],
        )
        os.makedirs(name=self.output_dir_name, exist_ok=True)
        output_video_name = self.input_filename[:-4] + "_output.mp4"

        # initialize the FourCC and a video writer object
        fourcc = cv.VideoWriter_fourcc("m", "p", "4", "v")
        self.video_writer = cv.VideoWriter(
            os.path.join(self.output_dir_name, output_video_name),
            fourcc,
            fps,
            (frame_width, frame_height),
        )
        return

    def initialize_output_csv(self):
        if self.output_dir_name is None:
            self.output_dir_name = os.path.join(
                self.settings_dict["output_directory"],
                dt.datetime.now().strftime("%d_%m_%y_%H-%M-%S_")
                + self.settings_dict["tag"],
            )
            os.makedirs(name=self.output_dir_name, exist_ok=True)

        output_csv_name = self.input_filename[:-4] + "_output.csv"

        self.csv_file = open(os.path.join(self.output_dir_name, output_csv_name), "w")
        self.csv_writer = csv.writer(self.csv_file)
        header = ["t", "frame number", "x", "y", "w", "h", "Classification", "ID"]
        self.csv_writer.writerow(header)

    def write_csv(self, detector):
        current_timestamp = self.start_datetime + dt.timedelta(
            seconds=float(self.frame_no) / self.fps
        )
        time_str = current_timestamp.strftime("%d-%m-%y_%H-%M-%S.%f")[:-3]

        rows = []
        if detector.current_objects is not None:
            for _, object_ in detector.current_objects.items():
                x, y, w, h = cv.boundingRect(object_.contours[-1])
                row = [
                    time_str,
                    f"{self.frame_no}",
                    f"{object_.midpoints[-1][0]}",
                    f"{object_.midpoints[-1][1]}",
                    str(w),
                    str(h),
                    f"{object_.classifications[-1]}",
                    f"{object_.ID}",
                ]
                rows.append(row)

        self.csv_writer.writerows(rows)

    def shutdown(self):
        self.video_cap.release()
        if "record_output_video" in self.settings_dict.keys():
            self.video_writer.release()
        cv.destroyAllWindows()

        if "record_output_csv" in self.settings_dict.keys():
            self.csv_file.close()

        if self.output_dir_name is not None:
            with open(os.path.join(self.output_dir_name, "settings.txt"), "w") as f:
                for key, setting in self.settings_dict.items():
                    f.write(f"{key}: {setting}")
                    f.write("\n")
            import zipfile

            filenames = [
                "algorithm/run_algorithm.py",
                "algorithm/FishDetector.py",
                "algorithm/InputOutputHandler.py",
                "algorithm/DetectedObject.py",
                "algorithm/visualization_functions.py",
            ]
            with zipfile.ZipFile(
                os.path.join(self.output_dir_name, "code.zip"), mode="w"
            ) as archive:
                for filename in filenames:
                    archive.write(filename)
