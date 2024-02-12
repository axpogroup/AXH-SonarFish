import argparse
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from azureml.core import Workspace
from dotenv import load_dotenv
from FishDetector import FishDetector
from InputOutputHandler import InputOutputHandler

from algorithm.DetectedObject import DetectedObject
from algorithm.validation import mot16_metrics

load_dotenv()


def read_ground_truth_into_dataframe(
    ground_truth_path: Path, filename: str
) -> pd.DataFrame:
    return pd.read_csv(Path(ground_truth_path) / Path(filename + "_ground_truth.csv"))


def extract_ground_truth_history(
    ground_truth_object_history, ground_truth, current_frame: int
):
    current_frame_df = ground_truth[ground_truth["frame"] == current_frame]
    for _, row in current_frame_df.iterrows():
        truth_detected = DetectedObject(
            identifier=row["id"],
            frame_number=row["frame"],
            contour=np.array(row[["x", "y", "w", "h"]]),
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
    workspace = Workspace(
        resource_group=os.getenv("RESOURCE_GROUP"),
        workspace_name=os.getenv("WORKSPACE_NAME"),
        subscription_id=os.getenv("SUBSCRIPTION_ID"),
    )
    mlflow.set_tracking_uri(workspace.get_mlflow_tracking_uri())
    experiment_name = settings_dict["experiment_name"]
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_params(settings_dict)
        if settings_dict.get("ground_truth_directory"):
            ground_truth_df = read_ground_truth_into_dataframe(
                ground_truth_path=Path(settings_dict["ground_truth_directory"]),
                filename=Path(settings_dict["file_name"]).stem,
            )
        else:
            ground_truth_df = pd.DataFrame()
        input_output_handler = InputOutputHandler(settings_dict)
        detector = FishDetector(settings_dict)
        object_history = {}
        truth_history = {}
        while input_output_handler.get_new_frame():
            detections, processed_frame_dict, runtimes = detector.detect_objects(
                input_output_handler.current_raw_frame
            )
            object_history = detector.associate_detections(detections, object_history)
            if not ground_truth_df.empty:
                truth_history = extract_ground_truth_history(
                    truth_history, ground_truth_df, input_output_handler.frame_no
                )
            else:
                truth_history = None
            input_output_handler.handle_output(
                processed_frame=processed_frame_dict,
                object_history=object_history,
                truth_history=truth_history,
                runtimes=runtimes,
                detector=detector,
            )

        if input_output_handler.output_csv_name is not None:
            df_detections = input_output_handler.get_detections_pd(object_history)
            df_detections = detector.classify_detections(df_detections)
            df_detections.to_csv(input_output_handler.output_csv_name, index=False)

        if settings_dict.get("ground_truth_directory"):
            file_name_prefix = Path(settings_dict["file_name"]).stem
            ground_truth_source = (
                Path(settings_dict["ground_truth_directory"])
                / f"{file_name_prefix}_ground_truth.csv"
            )
            test_source = (
                Path(settings_dict["output_directory"])
                / file_name_prefix
                / Path(file_name_prefix + ".csv")
            )
            ground_truth_source, test_source = (
                mot16_metrics.prepare_data_for_mot_metrics(
                    ground_truth_source, test_source
                )
            )
            mot16_metrics_dict = mot16_metrics.mot_metrics_enhanced_calculator(
                ground_truth_source, test_source
            )
            mlflow.log_metrics(mot16_metrics_dict)
