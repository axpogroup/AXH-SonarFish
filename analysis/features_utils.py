import copy
from typing import Optional

import cv2 as cv
import numpy as np
import pandas as pd
from numpy import ndarray


def calculate_features(measurements_df: pd.DataFrame, masks: dict[str, np.ndarray]) -> pd.DataFrame:
    measurements_df["binary_image"], measurements_df["tile_blob_counts"] = grayscale_to_binary(
        measurements_df["image_tile"]
    )
    feature_df = measurements_df.groupby("id").apply(lambda x: trace_window_metrics(x, masks))
    return measurements_df.join(feature_df, on="id", how="left")


def calculate_distance_between_starting_and_ending_point(detection):
    start_x, start_y = detection["x"].iloc[0], detection["y"].iloc[0]
    end_x, end_y = detection["x"].iloc[-1], detection["y"].iloc[-1]
    distance = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
    return distance


def calculate_average_distance_from_start(detection: pd.DataFrame) -> ndarray:
    start_x, start_y = detection["x"].iloc[0], detection["y"].iloc[0]
    distances = np.sqrt((detection["x"] - start_x) ** 2 + (detection["y"] - start_y) ** 2)
    return np.nanmean(distances)


def trace_window_metrics(detection: pd.DataFrame, masks: dict[str, np.array]) -> pd.Series:
    frame_diff = detection["frame"].iloc[-1] - detection["frame"].iloc[0]
    time_ratio_near_rake, dist_near_rake = calculate_rake_path_ratio(detection, masks["rake_mask"])
    return pd.Series(
        {
            "traversed_distance": sum_euclidean_distance_between_positions(detection),
            "frame_diff": frame_diff,
            "average_curvature": calculate_average_curvature(detection),
            "average_overlap_ratio": calculate_average_overlap_ratio(detection),
            "average_bbox_size": calculate_average_bbox_size(detection),
            "rake_time_ratio": time_ratio_near_rake,
            "dist_near_rake": dist_near_rake,
            "flow_area_time_ratio": calculate_flow_area_time_ratio(detection, masks["flow_area_mask"]),
            "average_distance_from_start": calculate_average_distance_from_start(detection),
            "average_contour_area": np.mean(detection["contour_area"]),
            "distance_between_starting_and_ending_point": calculate_distance_between_starting_and_ending_point(
                detection
            ),
            "max_blob_count": max_blob_count(detection),
        }
    )


def max_blob_count(detection: pd.DataFrame) -> int:
    return max(detection["tile_blob_counts"])


def calculate_average_bbox_size(group: pd.DataFrame) -> float:
    return np.mean(group["w"] * group["h"])


def calculate_rake_path_ratio(detection: pd.DataFrame, rake_mask: np.array) -> tuple[float, float]:
    x = detection["x"].values
    y = detection["y"].values
    is_near_rake = rake_mask[y.astype(int), x.astype(int)]
    time_ratio_near_rake = np.sum(is_near_rake) / len(is_near_rake)

    x_diff = np.diff(x[is_near_rake])
    y_diff = np.diff(y[is_near_rake])
    dist_near_rake = np.sum(np.sqrt(x_diff**2 + y_diff**2))
    return time_ratio_near_rake, dist_near_rake


def calculate_flow_area_time_ratio(detection: pd.DataFrame, flow_mask: np.array) -> float:
    x = detection["x"].values
    y = detection["y"].values
    is_in_flow = flow_mask[y.astype(int), x.astype(int)]
    time_ratio_in_flow = np.sum(is_in_flow) / len(is_in_flow)
    return time_ratio_in_flow


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
    curvature = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (dx_dt**2 + dy_dt**2 + 1e-8) ** (3 / 2)
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
    return np.mean(overlap_ratios[:-5])


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
    return get_ratio_of_overlaping_pixels(merged_img)


def show_overlap(merged_img):
    cv.namedWindow("Resized_Window", cv.WINDOW_NORMAL)
    cv.resizeWindow("Resized_Window", 500, 700)
    cv.imshow("Resized_Window", resize_img(merged_img))
    cv.waitKey()
    cv.destroyAllWindows()


def get_ratio_of_overlaping_pixels(merged_img):
    overlaping_pixels = np.count_nonzero(np.all(merged_img == (0, 200, 200), axis=2))
    number_of_non_overlaping_in_first = np.count_nonzero(np.all(merged_img == (0, 0, 200), axis=2))
    number_of_overlaping_in_second = np.count_nonzero(np.all(merged_img == (0, 200, 0), axis=2))
    if number_of_non_overlaping_in_first + number_of_overlaping_in_second != 0:
        ratio = overlaping_pixels / (number_of_non_overlaping_in_first + number_of_overlaping_in_second)
    elif overlaping_pixels != 0:
        ratio = overlaping_pixels
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


def grayscale_to_binary(
    detection: pd.Series,
    difference_threshold_scaler: float = 0.3,
) -> np.ndarray:
    binaries = []
    blob_counts = []
    for image in detection:
        try:
            adaptive_threshold = difference_threshold_scaler * cv.blur(image.astype("uint8"), (10, 10))
            image[np.abs(image) < adaptive_threshold] = 0
            image = (np.abs(image) + 127).astype("uint8")
            _, binary_image = cv.threshold(image, 127 + difference_threshold_scaler, 255, 0)
            binary_image = np.squeeze(binary_image)
            binary_image, img_blob_counts = remove_small_blobs(binary_image)
        except IndexError:
            binary_image = None
        binaries.append(binary_image)
        blob_counts.append(img_blob_counts)

    return binaries, blob_counts


def remove_small_blobs(
    binary_image: np.ndarray,
    min_blob_to_area_ratio: int = 4,
    min_blob_pixel_count: int = 5,
) -> tuple[np.ndarray, int]:
    # Invert the binary image
    inverted_image = 255 - binary_image

    # Find contours in the inverted image
    contours, _ = cv.findContours(inverted_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Calculate the area for each contour
    areas = [cv.contourArea(cnt) for cnt in contours]

    # Calculate the average area
    avg_area = np.mean(areas)

    # Create a new binary image
    new_binary_image = np.zeros_like(binary_image)
    blob_counts = 0
    for cnt in contours:
        if (
            cv.contourArea(cnt) > 1.0 / min_blob_to_area_ratio * avg_area
            and cv.contourArea(cnt) >= min_blob_pixel_count
        ):
            cv.drawContours(new_binary_image, [cnt], -1, 255, thickness=cv.FILLED)
            blob_counts += 1

    # Invert the new binary image back to its original state
    new_binary_image = 255 - new_binary_image

    return new_binary_image, blob_counts
