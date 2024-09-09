import datetime as dt
import sys
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.append(".")
from algorithm.DetectedObject import BoundingBox, KalmanTrackedBlob
from algorithm.FishDetector import FishDetector
from algorithm.InputOutputHandler import InputOutputHandler
from algorithm.settings import Settings
from algorithm.validation import mot16_metrics
from algorithm.visualization_functions import TRUTH_LABEL_NO

load_dotenv()


def read_labels_into_dataframe(labels_path: Path, labels_filename: str) -> Optional[pd.DataFrame]:
    labels_path = Path(labels_path) / labels_filename
    if labels_path.exists():
        print(f"Found labels file: {labels_path}")
    else:
        print(f"No labels file found at: {labels_path}")
        return None
    return pd.read_csv(labels_path)


def find_valid_previous_video(gap_seconds: int, settings: Settings):
    current_timestamp = InputOutputHandler.extract_timestamp_from_filename(
        settings.file_name, settings.file_timestamp_format
    )
    if current_timestamp is None:
        print("Could not extract timestamp from current video.")
        return None

    video_files = sorted(Path(settings.input_directory).glob("*.mp4"))
    closest_video = None

    min_time_difference = None
    for video_file in video_files:
        video_timestamp = InputOutputHandler.extract_timestamp_from_filename(
            video_file.name, settings.file_timestamp_format
        )
        if video_timestamp is None:
            continue
        time_difference = current_timestamp - video_timestamp
        if time_difference.total_seconds() <= 0:
            continue
        elif min_time_difference is None or abs(time_difference) < abs(min_time_difference):
            closest_video = video_file
            min_time_difference = time_difference

    # Check the duration of the closest video to make sure there is no gap
    if closest_video:
        duration = InputOutputHandler.get_video_duration(closest_video)
        end_timestamp = InputOutputHandler.extract_timestamp_from_filename(
            closest_video.name, settings.file_timestamp_format
        ) + dt.timedelta(seconds=duration)
        if abs(current_timestamp - end_timestamp) <= dt.timedelta(seconds=gap_seconds):
            print(
                print(
                    f"Closest video found: {closest_video.name}, "
                    f"time gap between recordings: {(current_timestamp - end_timestamp).total_seconds()} seconds."
                )
            )
            return closest_video.name
        else:
            print(
                f"Closest video {closest_video.name} ends {(current_timestamp - end_timestamp)} "
                f"before the current video, which is outside the tolerance ({gap_seconds} seconds)."
            )
            return None
    else:
        print("No previous video found.")
        return None


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


def compute_metrics(settings):
    if settings.ground_truth_directory:
        file_name_prefix = Path(settings.file_name).stem
        ground_truth_source = Path(settings.ground_truth_directory) / f"{file_name_prefix}_ground_truth.csv"
        test_source = Path(settings.output_directory) / Path(file_name_prefix + ".csv")
        ground_truth_source, test_source = mot16_metrics.prepare_data_for_mot_metrics(ground_truth_source, test_source)
        mot16_metrics_dict = mot16_metrics.mot_metrics_enhanced_calculator(ground_truth_source, test_source)
        return mot16_metrics_dict


def burn_in_algorithm_on_previous_video(settings: Settings, burn_in_file_name: str):
    burn_in_settings = deepcopy(settings)  # avoid changing the original settings
    burn_in_settings.file_name = burn_in_file_name
    burn_in_settings.record_output_video = False
    burn_in_settings.display_output_video = False
    burn_in_settings.verbosity = 0

    print(f"Starting algorithm burn-in on video: {burn_in_file_name}")
    input_output_handler = InputOutputHandler(burn_in_settings)
    burn_in_detector = FishDetector(burn_in_settings)
    input_output_handler.get_new_frame(start_at_frames_from_end=burn_in_settings.long_mean_frames + 1)
    while input_output_handler.get_new_frame():
        _, _, _ = burn_in_detector.detect_objects(input_output_handler.current_raw_frame)

    return burn_in_detector, burn_in_settings


def run_tracking_algorithm(detector: FishDetector, settings: Settings):
    labels_filename = (
        Path(settings.file_name).stem + settings.labels_file_suffix + settings.ground_truth_directory + ".csv"
    )
    labels_df = read_labels_into_dataframe(
        labels_path=settings.ground_truth_directory,
        labels_filename=labels_filename,
    )

    input_output_handler = InputOutputHandler(settings)

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
            feature_to_load=settings.feature_to_load,
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

    if settings.record_output_video and settings.compress_output_video:
        input_output_handler.compress_output_video()
        input_output_handler.delete_temp_output_dir()

    return input_output_handler.output_csv_name


def main_algorithm(settings: Settings):

    burn_in_settings = deepcopy(settings)
    previous_video = find_valid_previous_video(gap_seconds=5)

    if previous_video:
        try:
            burn_in_detector, burn_in_settings = burn_in_algorithm_on_previous_video(
                burn_in_settings, burn_in_file_name=previous_video
            )
            detector = FishDetector(burn_in_settings, init_detector=burn_in_detector)
        except AssertionError:
            print("Burn-in algorithm failed. Starting algorithm without burn-in on previous video. Should not happen.")
            detector = FishDetector(burn_in_settings)
    else:
        print("Starting algorithm without burn-in on previous video.")
        detector = FishDetector(settings)

    output_csv_name = run_tracking_algorithm(detector)

    return output_csv_name


if __name__ == "__main__":

    main_algorithm()
