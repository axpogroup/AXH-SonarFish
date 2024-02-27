import copy
from typing import Optional

import cv2 as cv
import numpy as np
import pandas as pd


def calculate_features(measurements_df: pd.DataFrame) -> pd.DataFrame:
    feature_df = measurements_df.groupby("id").apply(trace_window_metrics)
    return measurements_df.join(feature_df, on="id", how="left")


def trace_window_metrics(detection: pd.DataFrame) -> pd.Series:
    frame_diff = detection["frame"].iloc[-1] - detection["frame"].iloc[0]
    return pd.Series(
        {
            "traversed_distance": sum_euclidean_distance_between_positions(detection),
            "frame_diff": frame_diff,
            "average_curvature": calculate_average_curvature(detection),
            "average_overlap_ratio": calculate_average_overlap_ratio(detection),
        }
    )


def sum_euclidean_distance_between_positions(detection: pd.DataFrame):
    x_diff = np.diff(detection["x"])
    y_diff = np.diff(detection["y"])
    euclidean_distances = np.sqrt(x_diff**2 + y_diff**2)
    traversed_distance = np.sum(euclidean_distances)
    return traversed_distance


def calculate_average_curvature(detection: pd.DataFrame) -> float:
    if len(detection["x"]) < 2 or len(detection["y"]) < 2:
        print(f"Skipping detection {detection['id']} because it has too few points.")
        return 0
    dx_dt, dy_dt = np.gradient(detection["x"]), np.gradient(detection["y"])
    d2x_dt2, d2y_dt2 = np.gradient(dx_dt), np.gradient(dy_dt)
    curvature = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (dx_dt**2 + dy_dt**2) ** (3 / 2)
    avg_curvature = np.nanmean(curvature)
    return float(avg_curvature)


def calculate_average_overlap_ratio(detection: pd.DataFrame):
    overlap_ratios = []
    prev_img = np.array([])
    for row in detection.itertuples():
        img = format_image_tile(row)
        if len(prev_img) != 0 and len(img) != 0:
            img, prev_img = pad_images_to_have_same_shape(img, prev_img)
            overlap_ratios.append(get_overlap_ratio(prev_img, img))
        prev_img = img
    return np.mean(overlap_ratios)


# Code from https://stackoverflow.com/questions/74657074/find-new-blobs-comparing-two-different-binary-images
def get_overlap_ratio(img1, img2, visualize: Optional[bool] = None):
    # Replace 255 with 100 (we want the sum img1+img2 not to overflow)
    ret, img1 = cv.threshold(img1, 127, 255, cv.THRESH_BINARY)
    ret, img2 = cv.threshold(img2, 127, 255, cv.THRESH_BINARY)
    img1[img1 >= 128] = 100
    img2[img2 >= 128] = 100

    # Dilate both images - assume close blobs are the same blob
    # (two blobs are considered overlapped even if they are close but not tuching).
    dilated_img1 = cv.dilate(img1, np.ones((11, 11), np.uint8))
    dilated_img2 = cv.dilate(img2, np.ones((11, 11), np.uint8))

    # Sum two images - in the sum, the value of overlapping parts of blobs is going to be 200
    sum_img = dilated_img1 + dilated_img2

    cv.floodFill(sum_img, None, (0, 0), 0, loDiff=0, upDiff=0)  # Remove the white frame.

    # cv.imshow('sum_img before floodFill', sum_img)  # Show image for testing.

    # Find pixels with value 200 (the overlapping blobs).
    thesh = cv.threshold(sum_img, 199, 255, cv.THRESH_BINARY)[1]

    # Find contours (of overlapping blobs parts)
    cnts = cv.findContours(thesh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]

    # Iterate contours and fill the overlapping part, and the non-zero pixels around it (i.e with value 100) with zero.
    for c in cnts:
        area_tresh = 0  # Optional
        area = cv.contourArea(c)
        if area > area_tresh:  # Ignore very tiny contours
            x, y = tuple(c[0, 0])  # Get coordinates of first pixel in the contour
            if sum_img[y, x] == 200:
                cv.floodFill(
                    sum_img, None, (x, y), 0, loDiff=100, upDiff=0
                )  # Setting loDiff=100 is set for filling pixels=100 (and pixels=200)

    sum_img[sum_img == 200] = 0  # Remove the small remainders

    sum_img[(img1 == 0) & (dilated_img1 == 100)] = 0  # Remove dilated pixels from dilated_img1
    sum_img[(img2 == 0) & (dilated_img2 == 100)] = 0  # Remove dilated pixels from dilated_img2
    sum_img[(img1 == 100) & (img2 == 0)] = (
        0  # Remove all the blobs that are only in first image (assume new blobs are "bored" only in image2)
    )

    merged_img = cv.merge((sum_img * 2, img1 * 2, img2 * 2))

    # The output image is img1, without the
    output_image = img1.copy()
    output_image[sum_img == 100] = 0

    # Show images for testing.
    if visualize:
        show_overlap(merged_img)
    return get_ratio_of_yellow_pixels(merged_img)


def show_overlap(merged_img):
    cv.namedWindow("Resized_Window", cv.WINDOW_NORMAL)
    cv.resizeWindow("Resized_Window", 500, 700)
    cv.imshow("Resized_Window", resize_img(merged_img))
    cv.waitKey()
    cv.destroyAllWindows()


def get_ratio_of_yellow_pixels(merged_img):
    number_of_yellow_pixels = np.count_nonzero(np.all(merged_img == (0, 200, 200), axis=2))
    number_of_red_pixels = np.count_nonzero(np.all(merged_img == (0, 0, 200), axis=2))
    number_of_green_pixels = np.count_nonzero(np.all(merged_img == (0, 200, 0), axis=2))
    if number_of_red_pixels + number_of_green_pixels != 0:
        ratio = number_of_yellow_pixels / (number_of_red_pixels + number_of_green_pixels)
    elif number_of_yellow_pixels != 0:
        ratio = number_of_yellow_pixels
    else:
        ratio = 0
    return ratio


def resize_img(img):
    scale_percent = 1000  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)


def format_image_tile(row):
    drawing = copy.deepcopy(row.image_tile)
    if len(drawing.shape) == 3:
        drawing = drawing[0]
    img = np.ascontiguousarray(drawing, dtype=np.uint8)
    return img


def get_border_rect(
    img,
    max_height,
    max_width,
):
    height_diff = max_height - img.shape[0]
    width_diff1 = max_width - img.shape[1]
    top = height_diff // 2
    bottom = height_diff - top
    left = width_diff1 // 2
    right = width_diff1 - left
    return top, bottom, left, right


def pad_images_to_have_same_shape(img, prev_img):
    max_height = max(img.shape[0], prev_img.shape[0])
    max_width = max(img.shape[1], prev_img.shape[1])
    top1, bottom1, left1, right1 = get_border_rect(img, max_height, max_width)
    top2, bottom2, left2, right2 = get_border_rect(prev_img, max_height, max_width)
    prev_img = cv.copyMakeBorder(prev_img, top2, bottom2, left2, right2, cv.BORDER_CONSTANT, value=[0, 0, 0])
    img = cv.copyMakeBorder(img, top1, bottom1, left1, right1, cv.BORDER_CONSTANT, value=[0, 0, 0])
    return img, prev_img
