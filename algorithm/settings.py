from pydantic import BaseModel
from typing import List, Optional, Tuple
import yaml
from pathlib import Path
from pydantic_yaml import parse_yaml_raw_as, to_yaml_str
import argparse


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
    input_pixels_per_mm: float

    # Enhancement
    downsample: int
    contrast: int
    brightness: int
    long_mean_frames: int
    short_mean_frames: int
    difference_threshold_scaler: float
    dilation_kernel_mm: int
    median_filter_kernel_mm: int
    erosion_kernel_mm: int

    # Tracking
    tracking_method: str
    max_association_dist_mm: int
    phase_out_after_x_frames: int
    kalman_std_obj_initialization_factor: float
    kalman_std_obj_initialization_trace: List[float]
    kalman_std_process_noise_factor: float
    kalman_std_process_noise_trace: List[float]
    kalman_std_mmt_noise_factor: float
    kalman_std_mmt_noise_trace: List[float]
    kalman_rotate_mmt_noise_in_river_direction: bool
    kalman_trace_history_matching_budget: int
    kalman_max_iou_distance: float
    kalman_max_age: int
    kalman_n_init: int
    filter_nearest_neighbor: Optional[str]
    filter_blob_matching_metric: str
    filter_association_thresh: float
    filter_blob_elimination_metric: str
    bbox_size_to_stddev_ratio_threshold: int

    # Classification
    river_pixel_velocity: List[float]
    min_occurences_for_fish: int
    deviation_from_river_velocity: float

    # Visualization
    no_more_show_after_x_frames: int
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
