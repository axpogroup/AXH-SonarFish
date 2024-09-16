import argparse
import os
import sys
from pathlib import Path

import mlflow
import yaml
from azureml.core import Workspace
from dotenv import load_dotenv

sys.path.append(".")
from algorithm.DetectedObject import KalmanTrackedBlob
from algorithm.FishDetector import FishDetector
from algorithm.InputOutputHandler import InputOutputHandler
from algorithm.inputs import video_reading, labels
from algorithm.validation import mot16_metrics

load_dotenv()


def compute_metrics(settings_dict):
    if settings_dict.get("ground_truth_directory"):
        file_name_prefix = Path(settings_dict["file_name"]).stem
        ground_truth_source = Path(settings_dict["ground_truth_directory"]) / f"{file_name_prefix}_ground_truth.csv"
        test_source = Path(settings_dict["output_directory"]) / Path(file_name_prefix + ".csv")
        ground_truth_source, test_source = mot16_metrics.prepare_data_for_mot_metrics(ground_truth_source, test_source)
        mot16_metrics_dict = mot16_metrics.mot_metrics_enhanced_calculator(ground_truth_source, test_source)
        return mot16_metrics_dict


def burn_in_algorithm_on_previous_video(settings_dict: dict, burn_in_file_name: str):
    burn_in_settings = settings_dict.copy()
    burn_in_settings["file_name"] = burn_in_file_name
    burn_in_settings["record_output_video"] = False
    burn_in_settings["display_output_video"] = False
    burn_in_settings["verbosity"] = 0

    print(f"Starting algorithm burn-in on video: {burn_in_file_name}")
    input_output_handler = InputOutputHandler(burn_in_settings)
    burn_in_detector = FishDetector(burn_in_settings)
    input_output_handler.get_new_frame(start_at_frames_from_end=burn_in_settings.get("long_mean_frames", 0) + 1)
    while input_output_handler.get_new_frame():
        _, _, _ = burn_in_detector.detect_objects(input_output_handler.current_raw_frame)

    return burn_in_detector


def run_tracking_algorithm(settings_dict: dict, detector: FishDetector):
    labels_df = labels.read_labels_into_dataframe(
        labels_path=Path(settings_dict.get("ground_truth_directory", "")),
        labels_filename=Path(settings_dict["file_name"]).stem
        + settings_dict.get("labels_file_suffix", "_ground_truth")
        + ".csv",
    )
    input_output_handler = InputOutputHandler(settings_dict)

    object_history: dict[int, KalmanTrackedBlob] = {}
    label_history = {}
    print("Starting main algorithm.")
    while input_output_handler.get_new_frame():
        detections, processed_frame_dict, runtimes = detector.detect_objects(input_output_handler.current_raw_frame)
        object_history = detector.associate_detections(
            detections=detections, object_history=object_history, processed_frame_dict=processed_frame_dict
        )
        label_history = labels.extract_labels_history(
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
        df_detections = input_output_handler.get_detections_pd(object_history, detector=detector)
        df_detections = detector.classify_detections(df_detections)
        df_detections.to_csv(input_output_handler.output_csv_name, index=False)

    if settings_dict.get("record_output_video", False) and settings_dict.get("compress_output_video", False):
        input_output_handler.compress_output_video()
        input_output_handler.delete_temp_output_dir()

    return input_output_handler.output_csv_name


def main_algorithm(settings_dict: dict):
    previous_video = video_reading.find_valid_previous_video(settings_dict, gap_seconds=5)

    if previous_video:
        try:
            burn_in_detector = burn_in_algorithm_on_previous_video(settings_dict, burn_in_file_name=previous_video)
            detector = FishDetector(settings_dict, init_detector=burn_in_detector)
        except AssertionError:
            print("Burn-in algorithm failed. Starting algorithm without burn-in on previous video.")
            detector = FishDetector(settings_dict)
    else:
        print("Starting algorithm without burn-in on previous video.")
        detector = FishDetector(settings_dict)

    output_csv_name = run_tracking_algorithm(settings_dict, detector)

    return output_csv_name


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
