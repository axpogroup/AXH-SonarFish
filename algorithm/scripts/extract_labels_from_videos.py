# Code written by Leiv Andresen, HTD-A, leiv.andresen@axpo.com
import csv
import datetime
import datetime as dt
import glob
from pathlib import Path

import cv2 as cv
import yaml
from tqdm import tqdm

from algorithm.label_extraction.BoxDetector import BoxDetector
from algorithm.settings import Settings


def main(settings: Settings):
    filenames = glob.glob(settings.input_directory + "*.mp4")

    print("Found the following files: \n")
    latest_persistent_object_id = 250

    for file in filenames:
        file_name = Path(file).stem
        csv_path = Path(settings.csv_output_directory + file_name + settings.csv_output_suffix)
        if csv_path.exists() and not settings.overwrite_existing_csv:
            print(f"File {csv_path} already exists. Skipping.")
            continue
        print(f"\nProcessing  {file}")
        csv_file = open(csv_path.as_posix(), "w")
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
        detector = BoxDetector(settings, latest_persistent_object_id)
        frame_no = 0
        frames_total = int(video_cap.get(cv.CAP_PROP_FRAME_COUNT))
        fps = int(video_cap.get(cv.CAP_PROP_FPS))
        pbar = tqdm(total=frames_total, desc="Processing frames")
        while video_cap.isOpened():
            ret, raw_frame = video_cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Detection
            detector.process_frame(raw_frame)

            # Output
            current_timestamp = datetime.datetime.utcnow() + dt.timedelta(seconds=float(frame_no) / fps)
            csv_writer.writerows(
                detector.prepare_objects_for_csv(
                    timestr=current_timestamp.strftime("%y-%m-%d_%H-%M-%S.%f")[:-3],
                    file=file,
                )
            )
            # ... rest of the code ...

            pbar.update(1)
            frame_no += 1

        video_cap.release()
        if settings.csv_output_directory in settings.keys():
            csv_file.close()
        cv.destroyAllWindows()

        latest_persistent_object_id = detector.latest_persistent_object_id
        del detector
        pbar.close()


if __name__ == "__main__":
    with open("settings/tracking_box_settings.yaml") as f:
        settings = yaml.load(f, Loader=yaml.SafeLoader)
        print(settings)
        main(settings)
