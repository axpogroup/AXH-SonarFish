import csv
import datetime as dt
import glob
import os

import cv2 as cv
import numpy as np
import yaml
from BoxAndDotDetector import BoxAndDotDetector
from dateutil.relativedelta import relativedelta

# import code


def initialize_output_recording(input_video, output_video_file):
    # grab the width, height, fps and length of the video stream.
    frame_width = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(input_video.get(cv.CAP_PROP_FPS))

    # initialize the FourCC and a video writer object
    fourcc = cv.VideoWriter_fourcc("m", "p", "4", "v")
    return cv.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))


def parse_filename(filename):
    # filename = "2022-05-27_051500_771_2141 ZU BESPRECHEN passage am Schluss.mp4"
    date_part = "_".join(filename.split("_")[:2])
    start_frame = int(filename.split("_")[2])
    date_fmt = "%Y-%m-%d_%H%M%S"
    fps = 8.0
    start_dt = dt.datetime.strptime(date_part, date_fmt) + relativedelta(
        microseconds=(int((1 / fps) * 1000000 * start_frame))
    )
    suffix = " ".join(filename.split(" ")[1:])
    prefix = filename.split(" ")[0]

    return start_dt, prefix, suffix


"""
- Provide a folder with the video names. Exports of the same video need to have the same prefix
    (date, starting frame and ending frame number) and a space separating the prefix from the suffix of the filename.
- No objects must intersect. This will be detected and the affected frames
    of the video will be exported to a output video. If the output video is empty,
    then there was no intersection problems.
- If dots are visible from the beginning they will be recorded as appearing at the beginning of the frame,
    therefore they should appear in the course of the video. Once a dot appears it must remain in the image.
- The script will create csvs for each video and put the detections in it from every separate export,
    therefore nothing can occur twice in different exports of the same video.
- Each continous occurence of a box is denoted with a unique ID.
"""

if __name__ == "__main__":
    with open("settings/tracking_box_settings.yaml") as f:
        settings_dict = yaml.load(f, Loader=yaml.SafeLoader)
        print(settings_dict)

    os.makedirs(name=settings_dict["csv_output_directory"], exist_ok=True)
    os.makedirs(name=settings_dict["output_video_dir"], exist_ok=True)

    filenames = glob.glob(settings_dict["input_directory"] + "*.mp4")
    filenames.sort()
    print("Found the following files: \n")
    for file in filenames:
        print(file)
    print("\n")

    video_dt_csv_files = {}
    latest_persistent_object_id = 1

    for file in filenames:
        print(f"\nProcessing  {file}")
        current_file_dt, prefix, suffix = parse_filename(os.path.split(file)[-1])
        if prefix in video_dt_csv_files.keys():
            current_csv_file = video_dt_csv_files[prefix]
            csv_f = open(current_csv_file, "a")
            csv_writer = csv.writer(csv_f)
            print("Opening existing csv...")
        else:
            current_csv_file = settings_dict["csv_output_directory"] + prefix + ".csv"
            video_dt_csv_files[prefix] = current_csv_file
            csv_f = open(current_csv_file, "w")
            csv_writer = csv.writer(csv_f)
            header = [
                "t",
                "frame number",
                "x",
                "y",
                "w",
                "h",
                "Classification",
                "ID",
                "filename",
            ]
            csv_writer.writerow(header)
            print("Creating new csv...")

        video_cap = cv.VideoCapture(file)
        detector = BoxAndDotDetector(settings_dict, latest_persistent_object_id)

        if "output_video_dir" in settings_dict.keys():
            video_writer = initialize_output_recording(
                video_cap, (settings_dict["output_video_dir"] + os.path.split(file)[-1])
            )

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
            detector.process_frame(raw_frame, downsample=True)

            # Output
            current_timestamp = current_file_dt + dt.timedelta(
                seconds=float(frame_no) / fps
            )
            csv_writer.writerows(
                detector.prepare_objects_for_csv(
                    timestr=current_timestamp.strftime("%y-%m-%d_%H-%M-%S.%f")[:-3],
                    file=file,
                )
            )

            four_images = False
            fullres = True

            if four_images:
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

            elif fullres:
                disp = detector.draw_output(
                    detector.retrieve_frame(detector.current_raw, file),
                    debug=True,
                    classifications=True,
                    runtiming=True,
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

            # Video playback control
            if "output_video_dir" in settings_dict.keys() and detector.issue:
                print("Issue_detected.")
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
        if "output_video_dir" in settings_dict.keys():
            video_writer.release()
        if "csv_output_directory" in settings_dict.keys():
            csv_f.close()
        cv.destroyAllWindows()

        latest_persistent_object_id = detector.latest_persistent_object_id
        del detector
