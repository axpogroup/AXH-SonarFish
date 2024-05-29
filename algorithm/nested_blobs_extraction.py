import argparse
import copy
from pathlib import Path

import cv2 as cv
import numpy as np
import yaml

from algorithm.FishDetector import FishDetector
from analysis.features import load_csv_with_tiles


def mm_to_px(millimeters, settings):
    px = millimeters * settings["input_pixels_per_mm"] * settings["downsample"] / 100
    return px


def ceil_to_odd_int(number):
    number = int(np.ceil(number))
    return number + 1 if number % 2 == 0 else number


def further_extract_blobs(output_csv_df, settings):
    items_of_detection = {}
    for id, detection in output_csv_df.groupby("id"):
        if id not in items_of_detection.keys():
            items_of_detection[id] = {}
        for row in detection.itertuples():
            img = format_image_tile(row)
            if len(img) != 0:
                # cv.imwrite("test.jpg", img)
                image_blurred = cv.GaussianBlur(img, (5, 5), 0)
                adaptive_threshold = cv.adaptiveThreshold(
                    image_blurred,  # Input image
                    255,  # Maximum pixel value (white)
                    cv.ADAPTIVE_THRESH_GAUSSIAN_C,  # Adaptive thresholding method
                    cv.THRESH_BINARY,  # Thresholding type
                    21,  # Block size (size of the neighborhood area)
                    1,  # Constant subtracted from the mean (tune this parameter)
                )
                # cv.imwrite("test_thresh.jpg", adaptive_threshold)
                erosion_kernel_px = mm_to_px(settings["erosion_kernel_mm"], settings)
                kernel = cv.getStructuringElement(
                    cv.MORPH_ELLIPSE,
                    (
                        ceil_to_odd_int(erosion_kernel_px),
                        ceil_to_odd_int(erosion_kernel_px),
                    ),
                )
                eroded = cv.erode(adaptive_threshold, kernel, iterations=1)
                nested_detections, hierarchy = cv.findContours(eroded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                # for i in range(len(nested_detections)):
                #     cv.drawContours(eroded, nested_detections, -1, (0, 0, 0), 3)
                # # cv.imwrite("test_cont.jpg", eroded)
                items_of_detection[id][row.frame] = nested_detections
    for id in items_of_detection.keys():
        print(f"Detection {id} has: ")
        for frame in items_of_detection[id].keys():
            print(f"    {len( items_of_detection[id][frame])} contours in frame {frame}")


def format_image_tile(row):
    drawing = copy.deepcopy(row.image_tile)
    if len(drawing.shape) == 3:
        drawing = drawing[0]
    img = np.ascontiguousarray(drawing, dtype=np.uint8)
    return img


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
    output_csv_df = load_csv_with_tiles(
        Path(settings["output_directory"]) / Path(Path(settings["file_name"]).stem + ".csv")
    )
    detector = FishDetector(settings)
    further_extract_blobs(output_csv_df, settings)
