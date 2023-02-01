# Code written by Leiv Andresen, HTD-A, leiv.andresen@axpo.com

import cv2 as cv
import numpy as np
import math
import pandas as pd
import glob
import csv
import os
import datetime as dt
from dateutil.relativedelta import relativedelta


def parse_filename(filename):
    # filename = "2022-05-27_051500_771_2141 ZU BESPRECHEN passage am Schluss.mp4"
    # special cases "Video 12 2022-05-27_051500_771_2141 ZU BESPRECHEN passage am Schluss.mp4"
    # "Video 12 2022-05-27_051500_771_2141 ZU BESPRECHEN passage am Schluss.mp4"
    # "8_2021-11-02_051500_2219_4196 Rechenpass_FischROT"
    # "2022-05-27_214500_window#001 Rechenkontakte zu besprechen_Fisch"
    fps = 8
    if "Video 11_2022-06-16_230000_2437-3635 Abtast" in filename:
        filename_t = list(filename)
        filename_t[31] = '_'
        filename = ''.join(filename_t)

    if ("2021" in filename) or ("Video" in filename[:10]):
        date_part = "_".join(filename.split("_")[1:3])
        if "window#001" in filename:
            start_frame = 0
        else:
            start_frame = int(filename.split("_")[3])
        suffix = "_".join(filename.split("_")[5:])
        prefix = "_".join(filename.split("_")[0:4])
    elif "window#001" in filename:
        date_part = "_".join(filename.split("_")[:2])
        start_frame = 0
        suffix = " ".join(filename.split(" ")[1:])
        prefix = filename.split(" ")[0]
    else:
        date_part = "_".join(filename.split("_")[:2])
        start_frame = int(filename.split("_")[2])
        suffix = " ".join(filename.split(" ")[1:])
        prefix = filename.split(" ")[0]

    date_fmt = "%Y-%m-%d_%H%M%S"
    start_dt = dt.datetime.strptime(date_part, date_fmt) + relativedelta(
        microseconds=(int((1 / fps) * 1000000 * start_frame))
    )
    return start_dt, prefix, suffix


def calculate_transforms(dataframe_keypoints, transformations):
    for _, row in dataframe_keypoints.iterrows():
        delta_x, delta_y = row["point 2x"] - row["point 1x"], row["point 2y"] - row["point 1y"]

        # Assumes the two points are on a line parallel to the rake, possibly ADJUST
        angle = -math.atan(delta_y / delta_x)

        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

        if "_2021-" in row["filename"]:
            # the real world distance [mm] between the two points, possibly ADJUST
            s = 1400 / np.hypot(delta_x, delta_y)
        else:
            # the real world distance [mm] between the two points, possibly ADJUST
            s = 2168 / np.hypot(delta_x, delta_y)
        S = np.array([
            [s, 0, 0],
            [0, s, 0],
            [0, 0, 1]
        ])

        tx, ty = -row["point 1x"], -row["point 1y"]

        T = np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ])

        if "_2021-" in row["filename"]:
            # the real world position of point 1, possibly ADJUST
            t2x, t2y = 5000, 850
        else:
            # the real world position of point 1, possibly ADJUST
            t2x, t2y = 1912, 0

        T2 = np.array([
            [1, 0, t2x],
            [0, 1, t2y],
            [0, 0, 1]
        ])

        Iy = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])

        _, prefix, _ = parse_filename(os.path.split(row["filename"])[-1])
        if "_2021-" in row["filename"]:
            wasserhoehe = row["wasserhoehe"]
            jahr = 2021
        else:
            wasserhoehe = np.NAN
            jahr = 2022
        transformations[prefix] = {"R": R, "S": S, "T": T, "T2": T2, "Iy": Iy, "wasserhoehe": wasserhoehe, "y": jahr}

    return transformations


def transform_points(xy_old_homogenous, ts):
    points_new = ts["R"] @ ts["S"] @ ts["T"] @ xy_old_homogenous
    points_new = ts["Iy"] @ ts["T2"] @ points_new
    return points_new


if __name__ == "__main__":
    csv_files = glob.glob("output/csv/*.csv")
    csv_files.sort()

    print("Found the following files: \n")
    for file in csv_files:
        print(file)
    print("\n")

    transformations = {}
    # Specify the keypoint csv files, ADJUST
    # Wasserhöhe was added to the  2021 .csv file manually
    keypoints_2021 = pd.read_csv("output/keypoints_2021_mit_wasserhoehe.csv", delimiter=",")
    transformations = calculate_transforms(keypoints_2021, transformations)
    keypoints_2022 = pd.read_csv("output/keypoints_2022_new.csv", delimiter=",")
    transformations = calculate_transforms(keypoints_2022, transformations)

    print("Transformation prefixes:")
    for key in transformations.keys():
        print(key)

    output_directory = "output/csv_transformed/"
    summary_df = pd.DataFrame()
    for csv_file in csv_files:
        _, prefix, _ = parse_filename(os.path.split(csv_file)[-1][:-4])
        if prefix not in transformations.keys():
            # The following names were changed:
            # Video 7_2022-05-27_073000_window#001 zu besprechen abtasten + Rechenpassage.mp4
            # Video 6_2022-05-27_061500_window#001 Rechenkontakt.mp4
            print("missing ", prefix)
            continue

        print("Transforming ", prefix)

        ts = transformations[prefix]
        csv_frame = pd.read_csv(csv_file, delimiter=",")

        # Transform the points
        xy_old = csv_frame[["x - Koordinate", "y - Koordinate"]].to_numpy(dtype=float)
        xy_old_homogenous = np.vstack((xy_old.transpose(), np.ones(xy_old.shape[0])))
        xy_new_homogenous = transform_points(xy_old_homogenous, ts)

        xy_new = xy_new_homogenous[:2, :].transpose()
        csv_frame["x - Koordinate"] = xy_new[:, 0]
        csv_frame["y - Koordinate"] = xy_new[:, 1]
        csv_frame["x - Koordinate"] = csv_frame["x - Koordinate"].astype(int)
        csv_frame["y - Koordinate"] = csv_frame["y - Koordinate"].astype(int)
        csv_frame["Wasserhoehe"] = ts["wasserhoehe"]
        csv_frame["Jahr"] = ts["y"]

        csv_frame = csv_frame.drop(columns=["w - Breite", "h - Hoehe"])
        csv_frame = csv_frame[['Zeit', 'Framenummer', 'x - Koordinate', 'y - Koordinate', 'Klassifikation',
                               'ID', 'Wasserhoehe', 'Jahr', 'Dateiname']]
        csv_frame.to_csv((output_directory+prefix+"_transformiert.csv"))

        # Add to summary df
        summary_df = pd.concat([summary_df, csv_frame])

    summary_df.to_csv((output_directory+"alle_videos_transformiert.csv"))




