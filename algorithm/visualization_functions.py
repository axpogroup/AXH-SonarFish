import copy
from typing import Optional

import cv2 as cv
import numpy as np

from algorithm.DetectedObject import BoundingBox, DetectedBlob, KalmanTrackedBlob
from algorithm.FishDetector import FishDetector

FIRST_ROW = [
    "gray_boosted",
    "short_mean",
    "long_mean",
    "difference",
]
SECOND_ROW = ["difference_thresholded", "median_filter", "binary", "dilated"]
TRUTH_LABEL_NO = -1


def get_visual_output(
    object_history: dict[int, DetectedBlob],
    label_history: Optional[dict[int, BoundingBox]],
    detector: FishDetector,
    processed_frame: dict[str, np.ndarray],
    extensive=False,
    dual_output=False,
    color=(255, 200, 200),
    save_frame: str = "raw",
):
    assert not (dual_output and extensive), "dual_output and extensive can't both be True"

    if dual_output:
        # Get the raw frame without detections
        raw_frame = _retrieve_frame(save_frame, processed_frame)

        # Get the raw frame with detections
        detected_frame = _draw_detections_and_labels(
            detector=detector,
            object_history=object_history,
            label_history=label_history,
            processed_frame=_retrieve_frame(save_frame, processed_frame),
            color=color,
            paths=True,
            fullres=True,
        )

        # Concatenate the two frames horizontally
        disp = np.concatenate((raw_frame, detected_frame), axis=1)
    elif extensive:
        first_row_images = np.ndarray(shape=(270, 0, 3), dtype="uint8")
        second_row_images = np.ndarray(shape=(270, 0, 3), dtype="uint8")
        for frame_type in FIRST_ROW:
            first_row_images = np.concatenate(
                (
                    first_row_images,
                    _retrieve_frame(frame_type, processed_frame, puttext=frame_type),
                ),
                axis=1,
            )

        for frame_type in SECOND_ROW:
            second_row_images = np.concatenate(
                (
                    second_row_images,
                    _retrieve_frame(frame_type, processed_frame, puttext=frame_type),
                ),
                axis=1,
            )

        third_row_binary = _draw_detections_and_labels(
            object_history=object_history,
            label_history=label_history,
            detector=detector,
            processed_frame=_retrieve_frame("binary", processed_frame, puttext="detections"),
            paths=True,
            association_dist=True,
            color=color,
        )

        third_row_raw = _draw_detections_and_labels(
            object_history=object_history,
            label_history=label_history,
            detector=detector,
            processed_frame=_retrieve_frame("raw_downsampled", processed_frame, puttext="Final"),
            color=color,
        )

        third_row_images = np.concatenate(
            (
                third_row_raw,
                _retrieve_frame("internal_external", processed_frame, puttext="internal_external"),
                third_row_binary,
                _retrieve_frame("closed", processed_frame, puttext="closed"),
            ),
            axis=1,
        )
        disp = np.concatenate((first_row_images, second_row_images, third_row_images))

    else:
        img = _retrieve_frame(save_frame, processed_frame)
        if _draw_detections_and_labels:
            disp = _draw_detections_and_labels(
                detector=detector,
                object_history=object_history,
                label_history=label_history,
                processed_frame=_retrieve_frame("raw", processed_frame),
                color=color,
                paths=True,
                fullres=True,
            )
        else:
            disp = img

    return disp


def _draw_detections_and_labels(
    detector: FishDetector,
    object_history: dict[int, KalmanTrackedBlob],
    label_history: Optional[dict[int, BoundingBox]],
    processed_frame: dict[str, np.ndarray],
    color: tuple,
    **kwargs,
):
    disp = processed_frame
    if detector and detector.conf["show_detections"]:
        disp = _draw_detector_output(
            object_history,
            detector,
            processed_frame,
            color=color,
            **kwargs,
        )
    if label_history is not None:
        disp = _draw_labels(label_history, detector, disp, **kwargs)
    return disp


def _retrieve_frame(frame, frame_dict, puttext=None):
    if frame not in frame_dict.keys():
        img = None
    else:
        img = frame_dict[frame]

    out = copy.deepcopy(img)
    if out is None:
        out = np.zeros((270, 480, 3), dtype=np.uint8)

    if len(out.shape) != 3:
        out = cv.cvtColor(out, cv.COLOR_GRAY2BGR)

    if puttext is not None:
        cv.putText(
            out,
            puttext,
            (20, 20),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    return out


def _draw_labels(
    label_history: dict[int, BoundingBox],
    detector,
    img,
    paths=False,
    fullres=False,
    association_dist=False,
):
    labels_map = {0: "noise", 1: "fish", 2: "floating debris", -1: "truth"}
    for ID, obj in label_history.items():
        if is_detection_outdated(obj, detector):
            continue
        label = labels_map.get(obj.label)
        if label == "fish" or label == "truth":
            color = (57, 255, 20)
        elif label == "floating debris":
            color = (0, 255, 255)
        else:
            color = (57, 30, 255)
        draw_basic_bounding_box_and_path(association_dist, color, detector, fullres, img, obj, paths, label=label)
    return img


def _draw_detector_output(
    object_history: dict[int, KalmanTrackedBlob],
    detector,
    img,
    paths=False,
    fullres=False,
    association_dist=False,
    draw_detections: bool = True,
    annotate: bool = True,
    color=(255, 200, 200),
):
    for ID, obj in object_history.items():
        if is_detection_outdated(obj, detector) or obj.detection_is_tracked is False:
            continue
        h, scale, w, x, y = draw_basic_bounding_box_and_path(
            association_dist, color, detector, fullres, img, obj, paths
        )
        if draw_detections:
            if obj.ellipse_angles[-1] is not None and obj.ellipse_axes_lengths_pairs[-1] is not None:
                if not np.isnan(obj.ellipse_axes_lengths_pairs[-1]).any():
                    ellipse_axes_lengths = obj.ellipse_axes_lengths_pairs[-1].astype(int)
                    # if no nan in ellipse_axes_lengths:
                    cv.ellipse(
                        img,
                        (obj.midpoints[-1][0] * scale, obj.midpoints[-1][1] * scale),
                        (ellipse_axes_lengths[0] * scale, ellipse_axes_lengths[1] * scale),
                        obj.ellipse_angles[-1],
                        0,
                        360,
                        color,
                        1,
                    )
            text = ""
            if len(obj.means_of_pixels_intensity) > 0:
                text = f"ID:{obj.ID}"
                for feature in detector.conf.get("features_to_annotate", []):
                    if feature == "velocity":
                        if len(obj.velocities) > 0:
                            text += f", {feature}: {obj.velocities[-1] * scale}"
                    else:
                        text += f", {feature}: {obj.feature[feature]}"
            if detector.conf.get("annotate_detections", False):
                cv.putText(
                    img,
                    text,
                    (x - int(w / 2), y - int(h / 2) - 2 * scale),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
    return img


def draw_basic_bounding_box_and_path(
    association_dist, color, detector, fullres, img, obj, paths, label: Optional[str] = None
):
    if fullres:
        scale = int(100 / detector.conf["downsample"])
    else:
        scale = 1
    x, y = obj.midpoints[-1][0], obj.midpoints[-1][1]
    w, h = obj.bounding_boxes[-1][0], obj.bounding_boxes[-1][1]
    x, y, w, h = (
        x * scale,
        y * scale,
        w * scale,
        h * scale,
    )
    cv.rectangle(
        img,
        (x - int(w / 2), y - int(h / 2)),
        (x + int(w / 2), y + int(h / 2)),
        color,
        thickness=1,
    )
    if paths:
        for point in obj.midpoints:
            cv.circle(
                img,
                (point[0] * scale, point[1] * scale),
                1,
                color,
                thickness=-1,
            )
    if association_dist:
        cv.circle(
            img,
            (obj.midpoints[-1][0] * scale, obj.midpoints[-1][1] * scale),
            int(detector.mm_to_px(detector.conf["max_association_dist_mm"]) * scale),
            (0, 0, 255),
            1 * scale,
        )
    if label:
        if label == "fish" and obj.precalculated_feature is not None:
            label += f", {obj.precalculated_feature}"
        cv.putText(
            img,
            label,
            (x - int(w / 2), y - int(h / 2) - 2 * scale),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return h, scale, w, x, y


def draw_associations(associations, detections, object_history, img, color):
    for association in associations:
        cv.line(
            img,
            detections[association["detection_id"]].midpoints[-1],
            object_history[association["existing_object_id"]].midpoints[-1],
            color,
            2,
        )
        cv.putText(
            img,
            str(association["distance"]),
            detections[association["detection_id"]].midpoints[-1],
            cv.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
        )
    return img


def is_detection_outdated(obj, detector: Optional[FishDetector] = None):
    return detector.frame_number - obj.frames_observed[-1] > detector.conf["no_more_show_after_x_frames"]
