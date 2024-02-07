import datetime as dt
import os
from pathlib import Path

import cv2 as cv
import pandas as pd

import algorithm.visualization_functions as visualization_functions
from algorithm.DetectedObject import DetectedObject
from algorithm.utils import get_elapsed_ms


class InputOutputHandler:
    def __init__(self, settings_dict):
        self.video_writer = None
        self.settings_dict = settings_dict
        self.video_cap = cv.VideoCapture(
            Path(
                self.settings_dict["input_directory"] + self.settings_dict["file_name"]
            ).__str__()
        )
        self.input_filename = Path(self.settings_dict["file_name"])
        self.output_dir_name = None
        self.output_csv_name = None

        if (
            "record_output_csv" in settings_dict.keys()
            and self.settings_dict["record_output_csv"]
        ):
            if self.output_dir_name is None:
                self.output_dir_name = os.path.join(
                    self.settings_dict["output_directory"],
                    dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes")
                    + "_"
                    + self.settings_dict["tag"],
                )
                self.output_dir_name = self.output_dir_name.replace(":", "-")
                os.makedirs(name=self.output_dir_name, exist_ok=True)

            self.output_csv_name = os.path.join(
                self.output_dir_name, (self.input_filename.stem + ".csv")
            )

        self.playback_paused = False
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
    def get_detections_pd(object_history: dict[int, DetectedObject]) -> pd.DataFrame:
        rows = []
        for _, obj in object_history.items():
            for i in range(len(obj.frames_observed)):
                rows.append(
                    [
                        obj.frames_observed[i],
                        obj.ID,
                        obj.top_lefts_x[i],
                        obj.top_lefts_y[i],
                        obj.bounding_boxes[i][0],
                        obj.bounding_boxes[i][1],
                        obj.velocities[i][0],
                        obj.velocities[i][1],
                        obj.areas[i],
                    ]
                )

        return pd.DataFrame(
            rows,
            columns=[
                "frame",
                "id",
                "x",
                "y",
                "w",
                "h",
                "v_x",
                "v_y",
                "contour_area",
            ],
        )

    def trackbars(self, detector):
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

        if (
            "display_trackbars" in self.settings_dict.keys()
            and self.settings_dict["display_trackbars"]
        ):
            self.trackbars(detector)

        # Wait briefly for user input unless the video is paused
        if not self.playback_paused:
            self.usr_input = cv.waitKey(1)

        # If pause button was pressed, then the pause button can be used to playback frame by frame.
        # Any other button is pressed, playback will resume
        if self.usr_input == ord(" "):
            print("Press any key to continue playback or SPACE for next frame ... ")
            if cv.waitKey(0) == ord(" "):
                self.playback_paused = True
            else:
                self.playback_paused = False
            return

        if self.usr_input == 27:
            print("User pressed ESC, closing video stream ...")
            self.shutdown()
            return

    def handle_output(
        self, processed_frame, object_history, runtimes, detector, truth_history=None
    ):
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

        if ("display_output_video" in self.settings_dict.keys()) or \
            ("record_output_video" in self.settings_dict.keys()):
            if "display_mode_extensive" not in self.settings_dict.keys():
                extensive = False
            else: 
                extensive = self.settings_dict["display_mode_extensive"]
            disp = visualization_functions.get_visual_output(
                object_history=object_history,
                truth_history=truth_history,
                detector=detector,
                processed_frame=processed_frame,
                extensive=extensive,
                save_frame=self.settings_dict["record_processing_frame"],
                draw_detections=self.settings_dict["draw_detections_on_saved_video"],
            )

            if self.settings_dict["record_output_video"]:
                if not self.video_writer:
                    self.initialize_output_recording(
                        frame_width=disp.shape[1], 
                        frame_height=disp.shape[0],
                    )
                self.video_writer.write(disp)

            if (
                "display_output_video" in self.settings_dict.keys()
                and self.settings_dict["display_output_video"]
            ):
                self.show_image(disp, detector)

    def initialize_output_recording(
            self, 
            frame_width: int = None, 
            frame_height: int = None,
        ):
        # grab the width, height, fps and length of the video stream.
        frame_width = int(self.video_cap.get(cv.CAP_PROP_FRAME_WIDTH))  \
            if frame_width is None else frame_width
        frame_height = int(self.video_cap.get(cv.CAP_PROP_FRAME_HEIGHT)) \
            if frame_height is None else frame_height
        fps = int(self.video_cap.get(cv.CAP_PROP_FPS))
        if self.settings_dict['record_processing_frame'] != 'raw':
            fps = fps // 2

        self.output_dir_name = os.path.join(
            self.settings_dict["output_directory"],
            self.input_filename.stem,
            # + self.settings_dict["tag"],
        )
        self.output_dir_name = self.output_dir_name.replace(":", "-")
        os.makedirs(name=self.output_dir_name, exist_ok=True)
        output_video_name = f"{self.input_filename[:-4]}_{self.settings_dict['record_processing_frame']}_output.mp4"

        # initialize the FourCC and a video writer object
        fourcc = cv.VideoWriter_fourcc("m", "p", "4", "v")
        self.video_writer = cv.VideoWriter(
            os.path.join(self.output_dir_name, output_video_name),
            fourcc,
            fps,
            (frame_width, frame_height),
        )

    def get_video_output_settings(self):
        frame_width = int(self.video_cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.video_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.video_cap.get(cv.CAP_PROP_FPS))
        return fps, frame_height, frame_width

    def shutdown(self):
        self.video_cap.release()
        if "record_output_video" in self.settings_dict.keys():
            self.video_writer.release()
        cv.destroyAllWindows()

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
                "algorithm/utils.py",
            ]
            with zipfile.ZipFile(
                os.path.join(self.output_dir_name, "code.zip"), mode="w"
            ) as archive:
                for filename in filenames:
                    archive.write(filename)
