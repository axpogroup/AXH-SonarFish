import argparse
from typing import List, Optional

import yaml
from pydantic import BaseModel
from pydantic_yaml import parse_yaml_raw_as


class Settings(BaseModel):

    # Input / Output
    file_name: str
    input_directory: str
    mask_directory: str
    output_directory: str
    ground_truth_directory: str
    labels_file_suffix: str
    mask_file: str
    file_timestamp_format: str

    # Azure ML
    experiment_name: str
    track_azure_ml: bool

    tag: str
    record_output_video: bool
    compress_output_video: bool
    record_processing_frame: str
    draw_detections_on_saved_video: bool
    store_raw_image_patch: bool
    store_median_image_patch: bool
    show_detections: bool
    display_output_video: bool
    display_mode_extensive: bool
    display_trackbars: bool
    feature_to_load: str

    # Size
    input_pixels_per_mm: float = 0.0838

    # Enhancement
    downsample: int = 25
    contrast: int = 2
    brightness: int = 30
    long_mean_frames: int = 400
    short_mean_frames: int = 10
    difference_threshold_scaler: float = 0.30
    dilation_kernel_mm: int = 500
    median_filter_kernel_mm: int = 200
    erosion_kernel_mm: int = 100

    # Tracking
    tracking_method: str
    max_association_dist_mm: int = 500
    phase_out_after_x_frames: int = 30
    kalman_std_obj_initialization_factor: float = 0.1
    kalman_std_obj_initialization_trace: List[float] = [
        1,
        1,
        0.1,
        1,
        0.5,
        0.5,
        0.0001,
        0.5,
    ]  # [x, y, w, h, vx, vy, va, vh]
    kalman_std_process_noise_factor: float = 0.2
    kalman_std_process_noise_trace: List[float] = [
        1,
        1,
        0.1,
        1,
        0.05,
        0.05,
        0.0001,
        0.05,
    ]  # [x, y, w, h, vx, vy, va, vh]
    kalman_std_mmt_noise_factor: float = 0.5
    kalman_std_mmt_noise_trace: List[float] = [5, 1.7, 1, 0.05]  # [x, y, a, h]
    kalman_rotate_mmt_noise_in_river_direction: bool
    kalman_trace_history_matching_budget: int = 10
    kalman_max_iou_distance: float = 0.5  # basically (1 - iou_min)
    kalman_max_age: int = 35
    kalman_n_init: int = 35
    filter_nearest_neighbor: Optional[str]
    filter_blob_matching_metric: str
    filter_association_thresh: float = 0.3
    filter_blob_elimination_metric: str
    bbox_size_to_stddev_ratio_threshold: int = 100

    # Classification
    river_pixel_velocity: List[float] = [2.35, -0.9]  # At full resolution
    min_occurences_for_fish: int = 20
    deviation_from_river_velocity: float = 0.5

    # Visualization
    no_more_show_after_x_frames: int = 1
    video_colormap: str

    # Logging
    verbosity: int


argParser = argparse.ArgumentParser(description="Run the fish detection algorithm with a settings .yaml file.")
argParser.add_argument("-yf", "--yaml_file", help="path to the YAML settings file", required=True)
argParser.add_argument("-if", "--input_file", help="path to the input video file")

args = argParser.parse_args()

with open(args.yaml_file) as f:
    yml = yaml.load(f, Loader=yaml.SafeLoader)
    if args.input_file is not None:
        print("replacing input file.")
        yml["file_name"] = args.input_file


settings = parse_yaml_raw_as(Settings, str(yml))

if __name__ == "__main__":
    print(settings)
