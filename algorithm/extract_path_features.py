import argparse
from pathlib import Path
from typing import Iterable, List, Union

import numpy as np
import pandas as pd
import yaml
from numpy import ndarray


def calculate_derivatives(
    x: List[float], y: List[float]
) -> tuple[Union[ndarray, Iterable[ndarray]], Union[ndarray, Iterable[ndarray]]]:
    return np.gradient(x), np.gradient(y)


def calculate_average_curvature(detection) -> np.ndarray:
    if len(detection["x"].to_list()) < 2 or len(detection["y"].to_list()) < 2:
        print(f"Skipping detection {detection['id']} because it has too few points.")
        return 0
    dx_dt, dy_dt = calculate_derivatives(detection["x"].to_list(), detection["y"].to_list())
    d2x_dt2, d2y_dt2 = calculate_derivatives(dx_dt, dy_dt)
    curvature = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (dx_dt**2 + dy_dt**2) ** (3 / 2)
    avg_curvature = np.nanmean(curvature)
    print(f"Average Curvature: {avg_curvature}", detection["classification"].unique())
    return avg_curvature


def extract_path_features(settings: dict):
    output_csv_df = pd.read_csv(Path(settings["output_directory"]) / (Path(settings["file_name"]).stem + ".csv"))
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

    extract_path_features(settings)
