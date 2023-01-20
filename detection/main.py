import csv
import datetime as dt
import os

import cv2 as cv
import visualization_functions
import yaml
from FishDetector import FishDetector
from VideoHandler import VideoHandler


if __name__ == "__main__":
    with open("settings/machine_settings_recordings.yaml") as f:
        settings_dict = yaml.load(f, Loader=yaml.SafeLoader)
        print(settings_dict)

    video_handler = VideoHandler(settings_dict)
    detector = FishDetector(settings_dict)

    if "record_output_csv" in settings_dict.keys():
        csv_f = open(settings_dict["record_output_csv"], "w")
        csv_writer = csv.writer(csv_f)
        header = ["t", "frame number", "x", "y", "w", "h", "Classification"]
        csv_writer.writerow(header)

    date_fmt = "%y-%m-%d_start_%H-%M-%S_crop_swarms_single_2.mp4"
    start_datetime = dt.datetime.strptime(
        os.path.split(settings_dict["input_file"])[-1], date_fmt
    )

    while video_handler.get_new_frame():
        # Detection
        detector.process_frame(video_handler.current_raw_frame)

        # Output
        if "record_output_csv" in settings_dict.keys():
            current_timestamp = start_datetime + dt.timedelta(
                seconds=float(video_handler.frame_no) / video_handler.fps
            )
            csv_writer.writerows(
                detector.prepare_objects_for_csv(
                    timestr=current_timestamp.strftime("%y-%m-%d_%H-%M-%S.%f")[:-3]
                )
            )

        disp = visualization_functions.get_rich_output(detector, four_images=False)
        video_handler.show_image(disp, playback_controls=True)
        video_handler.write_image(disp)

    if "record_output_csv" in settings_dict.keys():
        csv_f.close()
    del detector
