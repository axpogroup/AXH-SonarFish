import argparse
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field


class Settings(BaseModel):

    # Preprocessing
    overwrite_existing_files: Optional[bool] = Field(default=False)
    overwrite_existing_csv: Optional[bool] = Field(default=False)

    # Downsampling
    target_fps: Optional[int] = Field(default=10)

    # Input / Output
    file_name: Optional[str] = None
    keys: Optional[str] = None
    input_directory: Optional[str] = Field(default=None)
    mask_directory: Optional[str] = None
    output_directory: Optional[str] = None
    ground_truth_directory: Optional[str] = None
    labels_file_suffix: Optional[str] = None
    mask_file: Optional[str] = None
    mask_directory: Optional[str] = None
    file_timestamp_format: Optional[str] = Field(default="%Y-%m-%dT%H-%M-%S.%f")
    output_csv_name: Optional[str] = Field(default=None)
    csv_output_directory: Optional[str] = None
    csv_output_suffix: Optional[str] = None

    # Azure ML
    experiment_name: Optional[str] = None
    track_azure_ml: Optional[bool] = None

    tag: Optional[str] = None
    record_output_video: bool = Field(default=False)
    compress_output_video: bool = Field(default=False)
    record_processing_frame: Optional[str] = None
    draw_detections_on_saved_video: bool = Field(default=False)
    store_raw_image_patch: Optional[bool] = None
    store_median_image_patch: Optional[bool] = None
    show_detections: Optional[bool] = None
    display_output_video: Optional[bool] = None
    display_mode_extensive: Optional[bool] = None
    display_trackbars: Optional[bool] = None
    feature_to_load: Optional[str] = Field(default=None)

    # Size
    input_pixels_per_mm: float = Field(default=0.0838)

    # Enhancement
    downsample: int = Field(default=25)
    contrast: int = Field(default=2)
    brightness: int = Field(default=30)
    long_mean_frames: int = Field(default=400)
    short_mean_frames: int = Field(default=10)
    difference_threshold_scaler: float = Field(default=0.30)
    dilation_kernel_mm: int = Field(default=500)
    median_filter_kernel_mm: int = Field(default=200)
    erosion_kernel_mm: int = Field(default=100)

    # Tracking
    tracking_method: Optional[str] = None
    max_association_dist_mm: int = Field(default=500)
    phase_out_after_x_frames: int = Field(default=30)
    kalman_std_obj_initialization_factor: float = Field(default=0.1)
    kalman_std_obj_initialization_trace: List[float] = Field(
        default_factory=lambda: [
            1,
            1,
            0.1,
            1,
            0.5,
            0.5,
            0.0001,
            0.5,
        ]
    )  # [x, y, w, h, vx, vy, va, vh]
    kalman_std_process_noise_factor: float = 0.2
    kalman_std_process_noise_trace: List[float] = Field(
        default_factory=lambda: [
            1,
            1,
            0.1,
            1,
            0.05,
            0.05,
            0.0001,
            0.05,
        ]
    )  # [x, y, w, h, vx, vy, va, vh]
    kalman_std_process_noise_factor: float = Field(default=0.5)
    kalman_std_mmt_noise_trace: List[float] = Field(default_factory=lambda: [5, 1.7, 1, 0.05])  # [x, y, a, h]
    kalman_rotate_mmt_noise_in_river_direction: Optional[bool] = None
    kalman_trace_history_matching_budget: float = Field(default=10)
    kalman_max_iou_distance: float = Field(default=0.5)  # basically (1 - iou_min)
    kalman_max_age: int = Field(default=35)
    kalman_n_init: int = Field(default=35)
    filter_nearest_neighbor: Optional[bool] = None
    filter_blob_matching_metric: Optional[str] = None
    filter_association_thresh: float = Field(default=0.3)
    filter_blob_elimination_metric: Optional[str] = None
    bbox_size_to_stddev_ratio_threshold: int = Field(default=100)

    four_images: bool = Field(default=False)
    fullres: bool = Field(default=True)
    downsample: int = Field(default=100)
    long_mean_frames: int = Field(default=120)
    current_mean_frames: int = Field(default=10)
    std_dev_threshold: float = Field(default=2)
    median_filter_kernel: int = Field(default=3)

    # Detection and Tracking
    blur_filter_kernel: int = Field(default=9)
    threshold_contours: int = Field(default=35)
    max_association_dist: int = Field(default=35)
    phase_out_after_x_frames: int = Field(default=9)
    min_occurences_in_last_x_frames: List[int] = Field(default_factory=lambda: [5, 10])

    # Classification
    river_pixel_velocity: List[float] = Field(default_factory=lambda: [-1.91, -0.85])
    rotation_rad: float = Field(default=-2.7229)

    # Classification
    river_pixel_velocity: List[float] = Field(default_factory=lambda: [2.35, -0.9])  # At full resolution
    min_occurences_for_fish: int = Field(default=20)
    deviation_from_river_velocity: float = Field(default=0.5)

    # Visualization
    no_more_show_after_x_frames: int = Field(default=1)
    video_colormap: Optional[str] = None

    # Logging
    verbosity: int = Field(default=1)


def main():
    argParser = argparse.ArgumentParser(description="Run the fish detection algorithm with a settings .yaml file.")
    argParser.add_argument("-yf", "--yaml_file", help="path to the YAML settings file", required=True)
    argParser.add_argument("-if", "--input_file", help="path to the input video file")

    args = argParser.parse_args()

    with open(args.yaml_file) as f:
        yml = yaml.load(f, Loader=yaml.SafeLoader)
        if args.input_file is not None:
            print("replacing input file.")
            yml["file_name"] = args.input_file


if __name__ == "__main__":
    main()
