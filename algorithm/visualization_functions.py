import copy
from typing import Dict, Optional

import cv2 as cv
import numpy as np

from algorithm.DetectedObject import DetectedObject
from algorithm.FishDetector import FishDetector

FIRST_ROW = [
    "gray_boosted",
    "short_mean",
    "long_mean",
    "difference",
]
SECOND_ROW = ["difference_thresholded", "median_filter", "binary", "dilated"]


def get_visual_output(
    object_history: Dict[int, DetectedObject],
    detector: FishDetector,
    processed_frame: Dict[str, np.ndarray],
    truth_history: Optional[Dict[int, DetectedObject]] = None,
    extensive=False,
    color=(255, 200, 200),
    truth_color=(57, 255, 20),
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

        third_row_binary = _draw_detections_and_truth(
            object_history=object_history,
            truth_history=truth_history,
            detector=detector,
            processed_frame=_retrieve_frame(
                "binary", processed_frame, puttext="detections"
            ),
            paths=True,
            association_dist=True,
            color=color,
            truth_color=truth_color,
        )

        third_row_raw = _draw_detections_and_truth(
            object_history=object_history,
            truth_history=truth_history,
            detector=detector,
            processed_frame=_retrieve_frame(
                "raw_downsampled", processed_frame, puttext="Final"
            ),
            truth_color=truth_color,
            color=color,
        )

        third_row_images = np.concatenate(
            (
                third_row_raw,
                _retrieve_frame(
                    "internal_external", processed_frame, puttext="internal_external"
                ),
                third_row_binary,
                _retrieve_frame("closed", processed_frame, puttext="closed"),
            ),
            axis=1,
        )
        disp = np.concatenate((first_row_images, second_row_images, third_row_images))

    else:
        disp = _draw_detections_and_truth(
            detector=detector,
            object_history=object_history,
            truth_history=truth_history,
            processed_frame=_retrieve_frame("raw", processed_frame),
            color=color,
            truth_color=truth_color,
            paths=True,
            fullres=True,
            association_dist=True,
            annotate="velocities",
        )

    return disp


def _draw_detections_and_truth(
    detector,
    object_history,
    processed_frame,
    color,
    truth_color,
    truth_history=None,
    **kwargs
):
    disp = _draw_detector_output(
        object_history, detector, processed_frame, color=color, **kwargs
    )
    if truth_history is not None:
        disp = _draw_detector_output(
            truth_history, detector, disp, color=truth_color, **kwargs
        )
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


def _draw_detector_output(
    object_history,
    detector,
    img,
    paths=False,
    fullres=False,
    association_dist=False,
    annotate=False,
    color=(255, 200, 200),
):
    for ID, obj in object_history.items():
        if (
            detector.frame_number - obj.frames_observed[-1]
            > detector.conf["no_more_show_after_x_frames"]
        ):
            continue

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
                int(
                    detector.mm_to_px(detector.conf["max_association_dist_mm"]) * scale
                ),
                (0, 0, 255),
                1 * scale,
            )

        if annotate:
            if annotate == "velocities":
                pass
                text = (
                    "v [px/frame]: "
                    + "{:.2f}".format(obj.velocities[-1][0] * scale)
                    + ", "
                    + "{:.2f}".format(obj.velocities[-1][1] * scale)
                )
            else:
                text = str(annotate)

            cv.putText(
                img,
                text,
                (x - int(w / 2), y - int(h / 2) - 2 * scale),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

    return img


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
