import argparse
from pathlib import Path

import pandas as pd
import yaml
from FishDetector import FishDetector
from InputOutputHandler import InputOutputHandler

from algorithm.DetectedObject import DetectedObject


def read_ground_truth_into_dataframe(ground_truth_path: Path, filename: str):
    return pd.read_csv(
        Path(ground_truth_path) / Path(filename + "_labels_ground_truth.csv")
    )


def extract_ground_truth_history(
    ground_truth_object_history, ground_truth, current_frame: int
):
    current_frame_df = ground_truth[ground_truth["frame"] == current_frame]
    for _, row in current_frame_df.iterrows():
        truth_detected = DetectedObject(
            identifier=row["id"],
            frame_number=row["frame"],
            x=row["x"],
            y=row["y"],
            w=row["w"],
            h=row["h"],
            area=None,
        )
        if row["id"] not in ground_truth_object_history:
            ground_truth_object_history[row["id"]] = truth_detected
        else:
            ground_truth_object_history[row["id"]].update_object(truth_detected)
    return ground_truth_object_history


if __name__ == "__main__":
    argParser = argparse.ArgumentParser(
        description="Run the fish detection algorithm with a settings .yaml file."
    )
    argParser.add_argument(
        "-yf", "--yaml_file", help="path to the YAML settings file", required=True
    )
    argParser.add_argument("-if", "--input_file", help="path to the input video file")

    args = argParser.parse_args()

    with open(args.yaml_file) as f:
        settings_dict = yaml.load(f, Loader=yaml.SafeLoader)
        if args.input_file is not None:
            print("replacing input file.")
            settings_dict["file_name"] = args.input_file

    ground_truth_df = read_ground_truth_into_dataframe(
        ground_truth_path=Path(settings_dict["ground_truth_directory"]),
        filename=Path(settings_dict["file_name"]).stem,
    )

    input_output_handler = InputOutputHandler(settings_dict)
    detector = FishDetector(settings_dict)
    object_history = {}
    truth_history = {}

    while input_output_handler.get_new_frame():
        detections, processed_frame_dict, runtimes = detector.detect_objects(
            input_output_handler.current_raw_frame
        )
        object_history = detector.associate_detections(detections, object_history)
        truth_history = extract_ground_truth_history(
            truth_history, ground_truth_df, input_output_handler.frame_no
        )
        input_output_handler.handle_output(
            processed_frame=processed_frame_dict,
            object_history=object_history,
            truth_history=truth_history,
            runtimes=runtimes,
            detector=detector,
        )

    if input_output_handler.output_csv_name is not None:
        detections = input_output_handler.get_detections_pd(object_history)
        detections = detector.classify_detections(detections)
        detections.to_csv(input_output_handler.output_csv_name, index=False)
