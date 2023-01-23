import csv
import datetime as dt
import os

import cv2 as cv
import visualization_functions


class InputOutputHandler:
    def __init__(self, settings_dict):
        self.csv_file = None
        self.csv_writer = None
        self.video_writer = None
        self.settings_dict = settings_dict
        self.video_cap = cv.VideoCapture(self.settings_dict["input_file"])
        self.input_filename = os.path.split(self.settings_dict["input_file"])[-1]
        self.output_dir_name = None

        if "record_output_video" in settings_dict.keys() and self.settings_dict["record_output_video"]:
            self.initialize_output_recording()

        if "record_output_csv" in settings_dict.keys() and self.settings_dict["record_output_csv"]:
            self.initialize_output_csv()

        try:
            date_fmt = "%y-%m-%d_start_%H-%M-%S.mp4"
            self.start_datetime = dt.datetime.strptime(
                os.path.split(settings_dict["input_file"])[-1], date_fmt
            )
        except ValueError:
            self.start_datetime = dt.datetime(year=2000, month=1, day=1)

        self.frame_by_frame = False
        self.usr_input = None
        self.frame_no = 0
        self.frames_total = int(self.video_cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.video_cap.get(cv.CAP_PROP_FPS))

        self.current_raw_frame = None

    def get_new_frame(self):
        if self.video_cap.isOpened():
            ret, self.current_raw_frame = self.video_cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                self.shutdown()
                return False

            self.frame_no += 1
            return True

        else:
            print("ERROR: Video Capturer is not open.")
            return False

    def show_image(self, img):
        cv.imshow("frame", img)

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

    def handle_output(self, detector):
        if self.frame_no % 20 == 0:
            print(f"Processed {self.frame_no / self.frames_total * 100} % of video.")

        if "record_output_csv" in self.settings_dict.keys():
            self.write_csv(detector)

        if ("display_output_video" in self.settings_dict.keys()) or (
            "record_output_video" in self.settings_dict.keys()
        ):
            rich_display = (
                False
                if "display_mode_rich" not in self.settings_dict.keys()
                else self.settings_dict["display_mode_rich"]
            )
            disp = visualization_functions.get_visual_output(
                detector, rich_display=rich_display
            )

            if "record_output_video" in self.settings_dict.keys():
                self.video_writer.write(disp)

            if (
                "display_output_video" in self.settings_dict.keys()
                and self.settings_dict["display_output_video"]
            ):
                self.show_image(disp)

    def initialize_output_recording(self):
        # grab the width, height, fps and length of the video stream.
        frame_width = int(self.video_cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.video_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.video_cap.get(cv.CAP_PROP_FPS))

        self.output_dir_name = os.path.join(self.settings_dict["output_directory"],
                                            dt.datetime.now().strftime("%y_%m_%d_%H-%M-%S_") +
                                            self.settings_dict["tag"])
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
            self.output_dir_name = os.path.join(self.settings_dict["output_directory"],
                                            dt.datetime.now().strftime("%y_%m_%d_%H-%M-%S_") +
                                            self.settings_dict["tag"])
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
        time_str = current_timestamp.strftime("%y-%m-%d_%H-%M-%S.%f")[:-3]

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
            with open(os.path.join(self.output_dir_name, 'settings.txt'), 'w') as f:
                for key, setting in self.settings_dict.items():
                    f.write(f"{key}: {setting}")
                    f.write('\n')
            import zipfile
            filenames = ["main.py", "FishDetector.py", "InputOutputHandler.py", "Object.py",
                         "visualization_functions.py"]
            with zipfile.ZipFile(os.path.join(self.output_dir_name, 'code.zip'), mode="w") as archive:
                for filename in filenames:
                    archive.write(filename)

