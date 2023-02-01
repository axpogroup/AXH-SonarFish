# Code written by Leiv Andresen, HTD-A, leiv.andresen@axpo.com

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
    # special cases "Video 12 2022-05-27_051500_771_2141 ZU BESPRECHEN passage am Schluss.mp4"
    # "Video 12 2022-05-27_051500_771_2141 ZU BESPRECHEN passage am Schluss.mp4"
    # "8_2021-11-02_051500_2219_4196 Rechenpass_FischROT"
    # "2022-05-27_214500_window#001 Rechenkontakte zu besprechen_Fisch"
    fps = 8
    if "Video 11_2022-06-16_230000_2437-3635 Abtast" in filename:
        filename_t = list(filename)
        filename_t[31] = '_'
        filename = ''.join(filename_t)

    if ("2021" in filename) or ("Video" in filename[:10]):
        date_part = "_".join(filename.split("_")[1:3])
        if "window#001" in filename:
            start_frame = 0
        else:
            start_frame = int(filename.split("_")[3])
        suffix = "_".join(filename.split("_")[5:])
        prefix = "_".join(filename.split("_")[0:4])
    elif "window#001" in filename:
        date_part = "_".join(filename.split("_")[:2])
        start_frame = 0
        suffix = " ".join(filename.split(" ")[1:])
        prefix = filename.split(" ")[0]
    else:
        date_part = "_".join(filename.split("_")[:2])
        start_frame = int(filename.split("_")[2])
        suffix = " ".join(filename.split(" ")[1:])
        prefix = filename.split(" ")[0]

    date_fmt = "%Y-%m-%d_%H%M%S"
    start_dt = dt.datetime.strptime(date_part, date_fmt) + relativedelta(
        microseconds=(int((1 / fps) * 1000000 * start_frame))
    )
    return start_dt, prefix, suffix


if __name__ == "__main__":
    # Specify the output folders, possibly ADJUST
    with open("settings/tracking_box_settings.yaml") as f:
        settings_dict = yaml.load(f, Loader=yaml.SafeLoader)
        print(settings_dict)

    os.makedirs(name=settings_dict["csv_output_directory"], exist_ok=True)
    os.makedirs(name=settings_dict["output_video_dir"], exist_ok=True)

    # Specify the input folders, possibly ADJUST
    filenames_2021 = glob.glob("ARIS_videos/2021_processed/*.mp4")
    filenames_2021.sort()
    filenames_2022 = glob.glob("ARIS_videos/2022_processed/*.mp4")
    filenames_2022.sort()
    filenames = filenames_2021 + filenames_2022

    # filenames = glob.glob("ARIS_videos/2022_reexport/*.mp4")
    # filenames.sort()
    print("Found the following files: \n")
    for file in filenames:
        current_file_dt, prefix, suffix = parse_filename(os.path.split(file)[-1])
        print(current_file_dt, prefix, suffix)
    print("\n")

    issues = []

    video_dt_csv_files = {}
    latest_persistent_object_id = 250

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
                "Zeit",
                "Framenummer",
                "x - Koordinate",
                "y - Koordinate",
                "w - Breite",
                "h - Hoehe",
                "Klassifikation",
                "ID",
                "Dateiname",
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

            # cv.imshow("frame", disp)
            # Video playback control
            if "output_video_dir" in settings_dict.keys() and detector.issue:
                cv.imshow("frame", disp)
                print("Issue_detected.", file, frame_no)
                issues.append((str(file) + " frame number: " + str(frame_no)))
                video_writer.write(disp)

            # if not frame_by_frame:
            #     usr_input = cv.waitKey(1)
            # # if usr_input == ord(" "):
            #     if cv.waitKey(0) == ord(" "):
            #         frame_by_frame = True
            #     else:
            #         frame_by_frame = False
            #     print("Press any key to continue ... ")
            # if usr_input == 27:
            #     break
            #
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

    for issue in issues:
        print(issue)
