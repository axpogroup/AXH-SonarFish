import copy
from typing import Optional

import cv2 as cv
import numpy as np

from algorithm.DetectedObject import BoundingBox, DetectedBlob, KalmanTrackedBlob
from algorithm.FishDetector import FishDetector
from algorithm.settings import Settings

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
    settings: Settings,
    extensive=False,
    color=(255, 200, 200),
    save_frame: str = "raw",
):
    if extensive:
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
            settings=settings,
            paths=True,
            association_dist=True,
            color=color,
        )

        third_row_raw = _draw_detections_and_labels(
            object_history=object_history,
            label_history=label_history,
            detector=detector,
            processed_frame=_retrieve_frame("raw_downsampled", processed_frame, puttext="Final"),
            settings=settings,
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
                settings=settings,
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
    settings: Settings,
    color: tuple,
    **kwargs,
):
    disp = processed_frame
    if settings.show_detections or settings.draw_detections_on_saved_video:
        disp = _draw_detector_output(
            object_history,
            detector,
            processed_frame,
            settings=settings,
            color=color,
            **kwargs,
        )
    if label_history is not None:
        disp = _draw_labels(label_history, detector, disp, settings=settings, **kwargs)
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
    settings: Settings,
    paths=False,
    fullres=False,
    association_dist=False,
):
    labels_map = {0: "noise", 1: "fish", 2: "floating debris", -1: "truth"}
    for ID, obj in label_history.items():
        if is_detection_outdated(obj, detector, settings):
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
    settings: Settings,
    paths=False,
    fullres=False,
    association_dist=False,
    annotate=True,
    color=(255, 200, 200),
):
    for ID, obj in object_history.items():
        if is_detection_outdated(obj, detector, settings) or obj.detection_is_tracked is False:
            continue
        h, scale, w, x, y = draw_basic_bounding_box_and_path(
            association_dist, color, detector, fullres, img, obj, paths, settings
        )
        if annotate:
            if obj.ellipse_angles[-1] is not None and obj.ellipse_axes_lengths_pairs[-1] is not None:
                if not np.isnan(obj.ellipse_axes_lengths_pairs[-1]).any():
                    ellipse_axes_lengths = obj.ellipse_axes_lengths_pairs[-1].astype(int)
                    # if no nan in ellipse_axes_lengths:
                    cv.ellipse(
                        img,
                        (obj.midpoints[-1][0], obj.midpoints[-1][1]),
                        (ellipse_axes_lengths[0], ellipse_axes_lengths[1]),
                        obj.ellipse_angles[-1],
                        0,
                        360,
                        color,
                        1 * scale,
                    )
            text = ""
            if len(obj.means_of_pixels_intensity) > 0:
                ratio = obj.feature["bbox_size_to_stddev_ratio"]
                text = f"ID:{obj.ID}, ratio: {int(ratio)}"
                if len(obj.velocities) > 100:
                    text += f", v [px/frame]: {obj.velocities[-1] * scale}"
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
    association_dist,
    color,
    detector,
    fullres,
    img,
    obj,
    paths,
    settings,
    label: Optional[str] = None,
):
    if fullres:
        scale = int(100 / settings.downsample)
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
        thickness=1 * scale,
    )
    if paths:
        for point in obj.midpoints:
            cv.circle(
                img,
                (point[0] * scale, point[1] * scale),
                1 * scale,
                color,
                thickness=-1,
            )
    if association_dist:
        cv.circle(
            img,
            (obj.midpoints[-1][0] * scale, obj.midpoints[-1][1] * scale),
            int(detector.mm_to_px(settings.max_association_dist_mm) * scale),
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


def is_detection_outdated(obj, settings, detector: Optional[FishDetector] = None):
    return detector.frame_number - obj.frames_observed[-1] > settings.no_more_show_after_x_frames
