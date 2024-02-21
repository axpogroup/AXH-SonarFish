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


def calculate_average_curvature(x: List[float], y: List[float]) -> np.ndarray:

    dx_dt, dy_dt = calculate_derivatives(x, y)
    d2x_dt2, d2y_dt2 = calculate_derivatives(dx_dt, dy_dt)
    curvature = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (dx_dt**2 + dy_dt**2) ** (3 / 2)
    return np.nanmean(
        curvature,
    )


def extract_path_features(settings: dict):
    output_csv = pd.read_csv(Path(settings["output_directory"]) / Path(Path(settings["file_name"]).stem + ".csv"))
    ids_of_detections = output_csv["id"].unique()
    fish_curvatures = []
    object_curvatures = []
    for ids_of_detection in ids_of_detections:
        detection = output_csv[output_csv["id"] == ids_of_detection]
        x_coords = detection["x"].tolist()
        y_coords = detection["y"].tolist()
        if len(x_coords) < 2:
            print(f"Skipping detection {ids_of_detection} because it has too few points.")
            continue
        avg_curvature = calculate_average_curvature(x_coords, y_coords)
        if detection["classification"].unique()[0] == "fish":
            fish_curvatures.append(avg_curvature)
        else:
            object_curvatures.append(avg_curvature)
        print(f"Average Curvature: {avg_curvature}", detection["classification"].unique())
    print(f"Average fish curvature: {np.nanmean(fish_curvatures)}")
    print(f"Average object curvature: {np.nanmean(object_curvatures)}")


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
