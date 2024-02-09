# Code written by Leiv Andresen, HTD-A, leiv.andresen@axpo.com
import csv
import datetime
import datetime as dt
import glob

import cv2 as cv
import numpy as np
import yaml
from label_extraction.BoxDetector import BoxDetector

if __name__ == "__main__":
    # Specify the output folders, possibly ADJUST
    with open("settings/tracking_box_settings.yaml") as f:
        settings_dict = yaml.load(f, Loader=yaml.SafeLoader)
        print(settings_dict)

    filenames = glob.glob(settings_dict["input_directory"] + "*.mp4")

    print("Found the following files: \n")
    issues = []
    video_dt_csv_files = {}
    latest_persistent_object_id = 250

    for file in filenames:
        print(f"\nProcessing  {file}")
        path_parts = file.split("/")
        file_name = path_parts[-1].split(".mp4")[0]
        csv_file = open(
            settings_dict["csv_output_directory"]
            + file_name
            + settings_dict["csv_output_suffix"],
            "w",
        )
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "frame",
                "id",
                "x",
                "y",
                "w",
                "h",
                "confidence",
                "x_3d",
                "y_3d",
                "z_3d",
            ]
        )
        video_cap = cv.VideoCapture(file)
        detector = BoxDetector(settings_dict, latest_persistent_object_id)
        frame_by_frame = False
        frame_no = 0
        frames_total = int(video_cap.get(cv.CAP_PROP_FRAME_COUNT))
        fps = int(video_cap.get(cv.CAP_PROP_FPS))
        while video_cap.isOpened():
            ret, raw_frame = video_cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Detection
            detector.process_frame(raw_frame)

            # Output
            current_timestamp = datetime.datetime.utcnow() + dt.timedelta(
                seconds=float(frame_no) / fps
            )
            csv_writer.writerows(
                detector.prepare_objects_for_csv(
                    timestr=current_timestamp.strftime("%y-%m-%d_%H-%M-%S.%f")[:-3],
                    file=file,
                )
            )
            if settings_dict["four_images"]:
                try:
                    up = np.concatenate(
                        (
                            detector.draw_output(
                                detector.retrieve_frame(
                                    detector.current_raw, puttext="raw"
                                ),
                                debug=True,
                                classifications=True,
                            ),
                            detector.retrieve_frame(
                                detector.current_blue, puttext="blue"
                            ),
                        ),
                        axis=1,
                    )
                    down = np.concatenate(
                        (
                            detector.retrieve_frame(
                                detector.current_red, puttext="red"
                            ),
                            detector.retrieve_frame(
                                detector.current_green, puttext="green"
                            ),
                        ),
                        axis=1,
                    )
                    disp = np.concatenate((up, down))
                    disp = detector.draw_output(disp, only_runtime=True, runtiming=True)
                except TypeError:  # ValueError:
                    disp = raw_frame

            elif settings_dict["fullres"]:
                disp = detector.draw_output(
                    detector.retrieve_frame(detector.current_raw, file),
                    debug=True,
                    runtiming=True,
                )

            else:
                disp = np.concatenate(
                    (
                        detector.draw_output(
                            detector.current_enhanced, debug=True, runtiming=True
                        ),
                        detector.draw_output(detector.current_raw, runtiming=True),
                    )
                )

            # Video playback control
            if frame_no % 20 == 0:
                print(f"Processed {frame_no/frames_total*100} % of video.")
                if frame_no / frames_total * 100 > 35:
                    pass
            frame_no += 1

        video_cap.release()
        if "csv_output_directory" in settings_dict.keys():
            csv_file.close()
        cv.destroyAllWindows()

        latest_persistent_object_id = detector.latest_persistent_object_id
        del detector

    for issue in issues:
        print(issue)
