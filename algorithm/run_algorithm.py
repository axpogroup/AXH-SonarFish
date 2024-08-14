import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import yaml
from azureml.core import Workspace
from dotenv import load_dotenv

sys.path.append(".")
from algorithm.DetectedObject import BoundingBox, KalmanTrackedBlob
from algorithm.FishDetector import FishDetector
from algorithm.InputOutputHandler import InputOutputHandler
from algorithm.validation import mot16_metrics
from algorithm.visualization_functions import TRUTH_LABEL_NO

load_dotenv()


def read_labels_into_dataframe(labels_path: Path, labels_filename: str) -> Optional[pd.DataFrame]:
    labels_path = Path(labels_path) / labels_filename
    if labels_path.exists():
        print(f"Found labels file: {labels_path}")
    else:
        print("No labels file found.")
        return None
    return pd.read_csv(labels_path)


def extract_labels_history(
    label_history: dict[int, BoundingBox],
    labels: Optional[pd.DataFrame],
    current_frame: int,
    down_sample_factor: int = 1,
    feature_to_load: Optional[str] = None,
) -> Optional[dict[int, BoundingBox]]:
    if labels is None:
        return None
    # current_frame_df = labels[labels["frame"] == int(current_frame * down_sample_factor)]
    current_frame_df = labels[labels["frame"] == current_frame]
    for _, row in current_frame_df.iterrows():
        truth_detected = BoundingBox(
            identifier=row["id"],
            frame_number=row["frame"],
            contour=np.array(row[["x", "y", "w", "h"]]),
            label=int(row.get("assigned_label") or row.get("classification_v2", TRUTH_LABEL_NO)),
            precalculated_feature=row.get(feature_to_load, None),
        )
        if row["id"] not in label_history:
            label_history[row["id"]] = truth_detected
        else:
            label_history[row["id"]].update_object(truth_detected)
    return label_history


def compute_metrics(settings_dict):
    if settings_dict.get("ground_truth_directory"):
        file_name_prefix = Path(settings_dict["file_name"]).stem
        ground_truth_source = Path(settings_dict["ground_truth_directory"]) / f"{file_name_prefix}_ground_truth.csv"
        test_source = Path(settings_dict["output_directory"]) / Path(file_name_prefix + ".csv")
        ground_truth_source, test_source = mot16_metrics.prepare_data_for_mot_metrics(ground_truth_source, test_source)
        mot16_metrics_dict = mot16_metrics.mot_metrics_enhanced_calculator(ground_truth_source, test_source)
        return mot16_metrics_dict


def main_algorithm(settings_dict: dict):
    labels_df = read_labels_into_dataframe(
        labels_path=Path(settings_dict.get("ground_truth_directory", "")),
        labels_filename=Path(settings_dict["file_name"]).stem
        + settings_dict.get("labels_file_suffix", "_ground_truth")
        + ".csv",
    )

    input_output_handler = InputOutputHandler(settings_dict)
    detector = FishDetector(settings_dict)
    object_history: dict[int, KalmanTrackedBlob] = {}
    label_history = {}
    print("Starting main algorithm.")
    while input_output_handler.get_new_frame():
        detections, processed_frame_dict, runtimes = detector.detect_objects(input_output_handler.current_raw_frame)
        object_history = detector.associate_detections(
            detections=detections, object_history=object_history, processed_frame_dict=processed_frame_dict
        )
        label_history = extract_labels_history(
            label_history,
            labels_df,
            input_output_handler.frame_no,
            down_sample_factor=input_output_handler.down_sample_factor,
            feature_to_load=settings_dict.get("feature_to_load"),
        )
        input_output_handler.handle_output(
            processed_frame=processed_frame_dict,
            object_history=object_history,
            label_history=label_history,
            runtimes=runtimes,
            detector=detector,
        )
    if input_output_handler.output_csv_name is not None:
        df_detections = input_output_handler.get_detections_pd(object_history)
        df_detections = detector.classify_detections(df_detections)
        df_detections.to_csv(input_output_handler.output_csv_name, index=False)

    return input_output_handler.output_csv_name


if __name__ == "__main__":
    argParser = argparse.ArgumentParser(description="Run the fish detection algorithm with a settings .yaml file.")
    argParser.add_argument("-yf", "--yaml_file", help="path to the YAML settings file", required=True)
    argParser.add_argument("-if", "--input_file", help="path to the input video file")

    args = argParser.parse_args()

    with open(args.yaml_file) as f:
        settings = yaml.load(f, Loader=yaml.SafeLoader)
        if args.input_file is not None:
            print("replacing input file.")
            settings["file_name"] = args.input_file

    main_algorithm(settings)

    if settings.get("track_azure_ml", False):
        workspace = Workspace(
            resource_group=os.getenv("RESOURCE_GROUP"),
            workspace_name=os.getenv("WORKSPACE_NAME"),
            subscription_id=os.getenv("SUBSCRIPTION_ID"),
        )
        mlflow.set_tracking_uri(workspace.get_mlflow_tracking_uri())
        experiment_name = settings["experiment_name"]
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            mlflow.log_params(settings)
            main_algorithm(settings)
            metrics = compute_metrics(settings)
            mlflow.log_metrics(metrics)
