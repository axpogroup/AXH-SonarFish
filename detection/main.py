import csv
import datetime as dt
import os

import cv2 as cv
import numpy as np
import yaml
from FishDetector import FishDetector


def initialize_output_recording(input_video, output_video_file):
    # grab the width, height, fps and length of the video stream.
    frame_width = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(input_video.get(cv.CAP_PROP_FPS))

    # initialize the FourCC and a video writer object
    fourcc = cv.VideoWriter_fourcc("m", "p", "4", "v")
    return cv.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))


if __name__ == "__main__":
    with open("settings/machine_settings_recordings.yaml") as f:
        settings_dict = yaml.load(f, Loader=yaml.SafeLoader)
        print(settings_dict)

    video_cap = cv.VideoCapture(settings_dict["input_file"])
    detector = FishDetector(settings_dict)

    if "record_output_video" in settings_dict.keys():
        video_writer = initialize_output_recording(
            video_cap, settings_dict["record_output_video"]
        )

    if "record_output_csv" in settings_dict.keys():
        csv_f = open(settings_dict["record_output_csv"], "w")
        csv_writer = csv.writer(csv_f)
        header = ["t", "frame number", "x", "y", "w", "h", "Classification"]
        csv_writer.writerow(header)

    frame_by_frame = False
    frame_no = 0
    frames_total = int(video_cap.get(cv.CAP_PROP_FRAME_COUNT))
    date_fmt = "%y-%m-%d_start_%H-%M-%S_crop_swarms_single_2.mp4"
    start_datetime = dt.datetime.strptime(
        os.path.split(settings_dict["input_file"])[-1], date_fmt
    )
    fps = int(video_cap.get(cv.CAP_PROP_FPS))
    while video_cap.isOpened():
        ret, raw_frame = video_cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Detection
        detector.process_frame(raw_frame, downsample=True)

        # Output
        if "record_output_csv" in settings_dict.keys():
            current_timestamp = start_datetime + dt.timedelta(
                seconds=float(frame_no) / fps
            )
            csv_writer.writerows(
                detector.prepare_objects_for_csv(
                    timestr=current_timestamp.strftime("%y-%m-%d_%H-%M-%S.%f")[:-3]
                )
            )

        four_images = True
        fullres = False
        if four_images:
            try:
                up = np.concatenate(
                    (
                        detector.retrieve_frame(
                            detector.current_enhanced, puttext="enhanced"
                        ),
                        detector.retrieve_frame(
                            detector.current_blurred_enhanced,
                            puttext="blurred enhanced",
                        ),
                    ),
                    axis=1,
                )
                down = np.concatenate(
                    (
                        detector.draw_output(
                            detector.retrieve_frame(
                                detector.current_raw, puttext="raw"
                            ),
                            debug=False,
                            classifications=True,
                        ),
                        detector.draw_output(
                            detector.retrieve_frame(
                                detector.current_threshold, puttext="thresholded"
                            ),
                            debug=True,
                        ),
                    ),
                    axis=1,
                )
                disp = np.concatenate((up, down))
                disp = detector.draw_output(
                    detector.resize_img(disp, 400), only_runtime=True, runtiming=True
                )
            except ValueError:
                disp = raw_frame

        elif fullres:
            disp = detector.draw_output(
                raw_frame, classifications=True, runtiming=True, fullres=True
            )

        else:
            disp = np.concatenate(
                (
                    detector.draw_output(
                        detector.current_enhanced, debug=True, runtiming=True
                    ),
                    detector.draw_output(
                        detector.current_raw, classifications=True, runtiming=True
                    ),
                )
            )
        cv.imshow("frame", disp)

        if "record_output_video" in settings_dict.keys():
            video_writer.write(disp)

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

        if frame_no % 20 == 0:
            print(f"Processed {frame_no/frames_total*100} % of video.")
            if frame_no / frames_total * 100 > 35:
                pass
        frame_no += 1

    video_cap.release()
    if "record_output_video" in settings_dict.keys():
        video_writer.release()
    if "record_output_csv" in settings_dict.keys():
        csv_f.close()
    cv.destroyAllWindows()
