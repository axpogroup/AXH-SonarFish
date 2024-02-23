import argparse
import json
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
import yaml

from algorithm.FishDetector import FishDetector


def calculate_average_curvature(detection) -> float:
    x_coordinates = detection["x"].to_list()
    y_coordinates = detection["y"].to_list()
    if len(x_coordinates) < 2 or len(y_coordinates) < 2:
        print(f"Skipping detection {detection['id']} because it has too few points.")
        return 0
    dx_dt, dy_dt = np.gradient(x_coordinates), np.gradient(y_coordinates)
    d2x_dt2, d2y_dt2 = np.gradient(dx_dt), np.gradient(dy_dt)
    curvature = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (dx_dt**2 + dy_dt**2) ** (3 / 2)
    avg_curvature = np.nanmean(curvature)
    print(f"Average Curvature: {avg_curvature}", detection["classification"].unique())
    return float(avg_curvature)


def load_output_csv(settings: dict) -> pd.DataFrame:
    output_csv_df = pd.read_csv(Path(settings["output_directory"]) / (Path(settings["file_name"]).stem + ".csv"))
    output_csv_df["image_tile"] = output_csv_df["image_tile"].apply(lambda x: np.array(json.loads(x)))
    return output_csv_df


def extract_path_features(output_csv_df: pd.DataFrame):
    avg_curvatures_of_detections_df = pd.DataFrame()

    avg_curvatures_of_detections_series = output_csv_df.groupby("id").apply(
        lambda detection: calculate_average_curvature(detection)
    )
    avg_curvatures_of_detections_df["id"] = avg_curvatures_of_detections_series.index
    avg_curvatures_of_detections_df["average_curvature"] = avg_curvatures_of_detections_series.values
    df_filtered_detections = output_csv_df.drop_duplicates(subset=["id"])[["id", "classification"]]
    merged = pd.merge(df_filtered_detections, avg_curvatures_of_detections_df, on="id")
    print("Average Curvatures of fish: ", merged[merged["classification"] == "fish"]["average_curvature"].mean())
    print("Average Curvatures of objects: ", merged[merged["classification"] == "object"]["average_curvature"].mean())


def mm_to_px(millimeters, settings):
    px = millimeters * settings["input_pixels_per_mm"] * settings["downsample"] / 100
    return px


def extract_displacement_vectors(output_csv_df, settings):
    items_of_detection = {}
    for id, detection in output_csv_df.groupby("id"):
        print(id)
        if id not in items_of_detection.keys():
            items_of_detection[id] = []
        for index, row in detection.iterrows():
            # _, im_bw = cv.threshold(row["image_tile"], 127 + settings["difference_threshold_scaler"], 255, 0)
            drawing = row["image_tile"]
            if len(drawing.shape) == 3:
                drawing = drawing[0]
            img = np.ascontiguousarray(drawing, dtype=np.uint8)
            if len(img) != 0:
                # cv.imwrite("test.jpg", img)
                image_blurred = cv.GaussianBlur(img, (5, 5), 0)
                adaptive_threshold = cv.adaptiveThreshold(
                    image_blurred,  # Input image
                    255,  # Maximum pixel value (white)
                    cv.ADAPTIVE_THRESH_GAUSSIAN_C,  # Adaptive thresholding method
                    cv.THRESH_BINARY,  # Thresholding type
                    11,  # Block size (size of the neighborhood area)
                    2,  # Constant subtracted from the mean (tune this parameter)
                )
                # cv.imwrite("test_thresh.jpg", adaptive_threshold)
                kernel = np.ones((3, 3), np.uint8)
                eroded = cv.erode(adaptive_threshold, kernel, iterations=1)
                nested_detections, hierarchy = cv.findContours(eroded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                # for i in range(len(nested_detections)):
                #     color = (0, 0, 0)
                #     cv.drawContours(img, nested_detections, i, color, 2, cv.LINE_8, hierarchy, 0)
                #     # Show in a window
                # # cv.imwrite("test_cont.jpg", img)
                items_of_detection[id].append((row["frame"], nested_detections))
    for id, detection in items_of_detection.items():
        print(f"Detection {id} has {len(detection)} frames.")


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
    output_csv_df = load_output_csv(settings)
    detector = FishDetector(settings)
    extract_displacement_vectors(output_csv_df, settings)
    # extract_path_features(output_csv_df)
