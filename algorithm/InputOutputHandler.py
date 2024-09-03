import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Optional

import cv2 as cv
import numpy as np
import pandas as pd

from algorithm import visualization_functions
from algorithm.DetectedObject import KalmanTrackedBlob
from algorithm.FishDetector import FishDetector
from algorithm.utils import get_elapsed_ms
from algorithm.settings import Settings

import subprocess


class InputOutputHandler:
    def __init__(
        self,
        settings: Settings,
    ):
        self.fps_out = 10
        self.video_writer = None
        self.__settings = settings
        self.input_filename = Path(self.__settings.file_name)
        self.set_video_cap()

        self.start_timestamp = self.extract_timestamp_from_filename(
            self.input_filename, self.__settings.file_timestamp_format
        )

        self.output_dir_name = self.__settings.output_directory
        self.temp_output_dir_name = Path.cwd() / "temp"
        self.temp_output_dir_name.mkdir(exist_ok=True)
        self.output_csv_name = Path(self.output_dir_name) / (self.input_filename.stem + ".csv")
        self.playback_paused = False
        self.usr_input = None
        self.frame_no = 0
        self.frames_total = int(self.video_cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.fps_in = int(self.video_cap.get(cv.CAP_PROP_FPS))
        self.start_ticks = 1
        self.index_in = -1
        self.index_out = -1
        self.down_sample_factor = self.fps_in / self.fps_out
        self.frame_retrieval_time = None
        self.last_output_time = None
        self.current_raw_frame = None


        def check_directories(self):
        # Check if the output directory exists
            if not Path(self.output_dir_name).exists():
                print(f"Error: Output directory {self.output_dir_name} does not exist.")
                return




    def get_new_frame(self, start_at_frames_from_end: int = 0) -> bool:
        start = cv.getTickCount()
        tries = 0

        if self.video_cap.isOpened():
            if start_at_frames_from_end > 0:
                # before we can set the index, we need to grab a frame successfully
                while tries < 5:
                    success = self.video_cap.grab()
                    if success:
                        break
                    tries += 1
                if not success:
                    print("Can't receive frame (stream end?). Exiting ...")
                    self.shutdown()
                    self.frame_retrieval_time = get_elapsed_ms(start)
                    return False

                tries = 0
                total_frames = int(self.video_cap.get(cv.CAP_PROP_FRAME_COUNT))
                start_frame = max(0, total_frames - start_at_frames_from_end)
                self.video_cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
                self.index_in = start_frame

            while tries < 5:
                success = self.video_cap.grab()
                if success:
                    self.index_in += 1
                    out_due = int(self.index_in / self.fps_in * self.fps_out)
                    if out_due > self.index_out:
                        success, frame = self.video_cap.retrieve()
                        if success:
                            self.index_out += 1
                            self.current_raw_frame = frame
                            self.frame_no += 1
                            self.frame_retrieval_time = get_elapsed_ms(start)
                            return True
                else:
                    tries += 1
            print("Can't receive frame (stream end?). Exiting ...")
            self.shutdown()
            self.frame_retrieval_time = get_elapsed_ms(start)
            return False

        else:
            print("ERROR: Video Capturer is not open.")
            self.frame_retrieval_time = get_elapsed_ms(start)
            return False

    def set_video_cap(self):
        input_file_path = Path(self.__settings.input_directory) / self.__settings.file_name
        assert input_file_path.exists(), f"Error: Input file {input_file_path} does not exist."

        self.video_cap = cv.VideoCapture(str(input_file_path))
        assert self.video_cap.isOpened(), "Error: Video Capturer could not be opened."

    @staticmethod
    def get_detections_pd(object_history: dict[int, KalmanTrackedBlob], detector: FishDetector) -> pd.DataFrame:
        rows = []
        for _, obj in object_history.items():
            if obj.detection_is_tracked:
                for i in range(len(obj.frames_observed)):
                    rows.append(
                        [
                            obj.frames_observed[i],
                            obj.ID,
                            obj.top_lefts_x[i],
                            obj.top_lefts_y[i],
                            obj.bounding_boxes[i][0],
                            obj.bounding_boxes[i][1],
                            obj.velocities[i][0] if len(obj.velocities) > i else np.nan,
                            obj.velocities[i][1] if len(obj.velocities) > i else np.nan,
                            obj.areas[i],
                            np.array(obj.feature_patch[i]),
                            # np.array(obj.raw_image_patch[i]),
                            obj.stddevs_of_pixels_intensity[i],
                        ]
                    )

        detections_df = pd.DataFrame(
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
                "image_tile",
                # "raw_image_tile",
                "stddev_of_intensity",
            ],
        )
        detections_df["image_tile"] = detections_df["image_tile"].apply(lambda x: json.dumps(x.tolist()))
        detections_df["burn_in_video"] = detector.burn_in_video_name
        # detections_df["raw_image_tile"] = detections_df["raw_image_tile"].apply(lambda x: json.dumps(x.tolist()))
        return detections_df

    def trackbars(self, detector):
        def change_current_mean_frames(value):
            if value == 0:
                value = 1
            self.__settings.short_mean_frames = value

        def change_long_mean_frames(value):
            if value < detector.current_mean_frames:
                self.__settings.long_mean_frames = self.__settings.short_mean_frames
            else:
                self.__settings.long_mean_frames = value

        def change_alpha(value):
            self.__settings.contrast = float(value) / 10

        def change_beta(value):
            self.__settings.brightness = value

        def change_diff_thresh(value):
            self.__settings.difference_threshold_scaler = value / 10

        def change_median_filter_kernel(value):
            self.__settings.median_filter_kernel_mm = value

        def change_dilatation_kernel(value):
            self.__settings.dilation_kernel_mm = value

        cv.createTrackbar(
            "contrast*10",
            "frame",
            int(self.__settings.contrast * 10),
            30,
            change_alpha,
        )
        cv.createTrackbar("brightness", "frame", self.__settings.brightness, 120, change_beta)
        cv.createTrackbar(
            "s_mean",
            "frame",
            self.__settings.short_mean_frames,
            120,
            change_current_mean_frames,
        )
        cv.createTrackbar(
            "l_mean",
            "frame",
            self.__settings.long_mean_frames,
            1200,
            change_long_mean_frames,
        )
        cv.createTrackbar(
            "diff_thresh*10",
            "frame",
            int(self.__settings.difference_threshold_scaler * 10),
            127,
            change_diff_thresh,
        )
        cv.createTrackbar(
            "median_f",
            "frame",
            self.__settings.median_filter_kernel_mm,
            1200,
            change_median_filter_kernel,
        )
        cv.createTrackbar(
            "dilate",
            "frame",
            self.__settings.dilation_kernel_mm,
            1200,
            change_dilatation_kernel,
        )

    def show_image(self, img, detector):
        cv.imshow("frame", img)

        if self.__settings.display_trackbars:
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
        self,
        processed_frame,
        object_history: dict[int, KalmanTrackedBlob],
        runtimes: Optional[dict[str, float]],
        detector,
        label_history=None,
    ):
        total_runtime, total_time_per_frame = self.calculate_total_time()
        if self.frame_no % 20 == 0 and self.__settings.verbosity > 1:
            if total_time_per_frame == 0:
                total_time_per_frame = 1
            print(
                f"Processed {'{:.1f}'.format(self.frame_no * self.down_sample_factor / self.frames_total * 100)} \
                    % of video."
            )
            if runtimes:
                print(
                    f"Runtimes [ms]: getFrame: {self.frame_retrieval_time} | Enhance: {runtimes['enhance']} | "
                    f"DetectTrack: {runtimes['detection_tracking']} | "
                    f"Total: {total_time_per_frame} | FPS: {'{:.1f}'.format(self.frame_no/(2*total_runtime/1000))}"
                )
        if self.__settings.display_output_video or self.__settings.record_output_video:
            
            disp = visualization_functions.get_visual_output(
                object_history=object_history,
                label_history=label_history,
                detector=detector,
                processed_frame=processed_frame,
                extensive = self.__settings.display_mode_extensive,
                save_frame=self.__settings.record_processing_frame,
            )

            # Put timestamp on frame
            timestamp = self.start_timestamp + dt.timedelta(seconds=self.index_in / self.fps_in)
            text_location = (int((0.74 * disp.shape[1])), int((0.907 * disp.shape[0])))
            cv.putText(
                disp,
                timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                text_location,
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            if self.__settings.record_output_video:
                if not self.video_writer:
                    self.initialize_output_recording(
                        frame_width=disp.shape[1],
                        frame_height=disp.shape[0],
                    )
                self.video_writer.write(disp)

            if self.__settings.display_output_video:
                self.show_image(disp, detector)

    def calculate_total_time(self):
        if self.last_output_time is not None:
            total_time_per_frame = get_elapsed_ms(self.last_output_time)
        else:
            self.start_ticks = cv.getTickCount()
            total_time_per_frame = 0
        total_runtime = get_elapsed_ms(self.start_ticks)
        self.last_output_time = cv.getTickCount()
        return total_runtime, total_time_per_frame

    def initialize_output_recording(
        self,
        frame_width: int = None,
        frame_height: int = None,
    ):
        # grab the width, height, fps and length of the video stream.
        frame_width = int(self.video_cap.get(cv.CAP_PROP_FRAME_WIDTH)) if frame_width is None else frame_width
        frame_height = int(self.video_cap.get(cv.CAP_PROP_FRAME_HEIGHT)) if frame_height is None else frame_height
        fps = self.fps_in
        if self.__settings.record_processing_frame != "raw":
            fps = fps // 2

        output_video_name = f"{self.input_filename.stem}_{self.__settings.record_processing_frame}_output.mp4"
        if self.__settings.compress_output_video:
            self.output_video_path = Path(self.temp_output_dir_name) / output_video_name
        else:
            self.output_video_path = Path(self.output_dir_name) / output_video_name

        # initialize the FourCC and a video writer object
        fourcc = cv.VideoWriter_fourcc("m", "p", "4", "v")
        self.video_writer = cv.VideoWriter(
            str(self.output_video_path),
            fourcc,
            fps,
            (frame_width, frame_height),
        )

    def compress_output_video(self):
        compressed_output_video_path = Path(self.output_dir_name) / self.output_video_path.name
        command = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-i",
            str(self.output_video_path),
            "-vf",
            "format=yuv420p",
            "-c:v",
            "libx264",
            "-crf",
            str(35),  # Constant Rate Factor (0-51, 0 is lossless)
            "-preset",
            "medium",
            str(compressed_output_video_path),
        ]

        




        print(f"Compressing output video to {compressed_output_video_path} ...")
        subprocess.run(command)

    def delete_temp_output_dir(self):
        if self.temp_output_dir_name.exists():
            print("Deleting temporary output directory ...")
            for file in self.temp_output_dir_name.iterdir():
                file.unlink()
            self.temp_output_dir_name.rmdir()

    def shutdown(self):
        self.video_cap.release()
        if self.__settings.record_output_video:
            self.video_writer.release()
        cv.destroyAllWindows()

    @staticmethod
    def extract_timestamp_from_filename(filename: str, file_timestamp_format: str) -> Optional[dt.datetime]:
        try:
            if file_timestamp_format == "start_%Y-%m-%dT%H-%M-%S.%f%z.mp4":
                return dt.datetime.strptime(str(Path(filename).stem[:-6]), "start_%Y-%m-%dT%H-%M-%S.%f")
            else:
                return dt.datetime.strptime(str(Path(filename)), file_timestamp_format)
        except Exception as e:
            print(f"{e}")
            return None

    @staticmethod
    def get_video_duration(filepath: Path):
        video = cv.VideoCapture(str(filepath))
        frames = video.get(cv.CAP_PROP_FRAME_COUNT)
        fps = video.get(cv.CAP_PROP_FPS)
        try:
            duration = frames / fps
        except ZeroDivisionError:
            duration = 0
        video.release()
        return duration
