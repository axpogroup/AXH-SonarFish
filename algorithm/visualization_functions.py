import copy
from typing import Optional

import cv2 as cv
import numpy as np

from algorithm.DetectedObject import BoundingBox, DetectedBoundingBox
from algorithm.FishDetector import FishDetector

FIRST_ROW = [
    "gray_boosted",
    "short_mean",
    "long_mean",
    "difference",
]
SECOND_ROW = ["difference_thresholded", "median_filter", "binary", "dilated"]


def get_visual_output(
    object_history: dict[int, DetectedBoundingBox],
    label_history: Optional[dict[int, BoundingBox]],
    detector: FishDetector,
    processed_frame: dict[str, np.ndarray],
    extensive=False,
    color=(255, 200, 200),
    truth_color=(57, 255, 20),
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
            paths=True,
            association_dist=True,
            color=color,
            truth_color=truth_color,
        )

        third_row_raw = _draw_detections_and_labels(
            object_history=object_history,
            label_history=label_history,
            detector=detector,
            processed_frame=_retrieve_frame("raw_downsampled", processed_frame, puttext="Final"),
            truth_color=truth_color,
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
                truth_color=truth_color,
                paths=True,
                fullres=True,
                annotate="velocities",
            )
        else:
            disp = img

    return disp


def _draw_detections_and_labels(
    detector: FishDetector,
    object_history: dict[int, DetectedBoundingBox],
    label_history: Optional[dict[int, BoundingBox]],
    processed_frame: dict[str, np.ndarray],
    color: tuple,
    truth_color: tuple,
    **kwargs,
):
    disp = _draw_detector_output(object_history, detector, processed_frame, color=color, **kwargs)
    if label_history is not None:
        disp = _draw_detector_output(label_history, detector, disp, annotate=False, color=truth_color, **kwargs)
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
    annotate=True,
    color=(255, 200, 200),
):
    for ID, obj in object_history.items():
        if is_detection_outdated_or_not_confirmed(obj, detector):
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
        # if obj.ellipse_angles[-1] is not None and obj.ellipse_axes_lengths_pairs[-1] is not None:
        #     cv.ellipse(
        #         img,
        #         (obj.midpoints[-1][0], obj.midpoints[-1][1]),
        #         (int(obj.ellipse_axes_lengths_pairs[-1][0]), int(obj.ellipse_axes_lengths_pairs[-1][1])),
        #         obj.ellipse_angles[-1],
        #         0,
        #         360,
        #         color,
        #         1 * scale,
        #     )
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
                int(detector.mm_to_px(detector.conf["max_association_dist_mm"]) * scale),
                (0, 0, 255),
                1 * scale,
            )
        if annotate:
            text = ""
            if len(obj.means_of_pixels_intensity) > 0:
                ratio = obj.bbox_size_to_stddev_ratio
                text = f"ID:{obj.ID}, ratio: {ratio}"
                if len(obj.velocities) > 100:
                    text += (
                        ", v [px/frame]: "
                        + str(obj.velocities[-1][0] * scale)
                        + ", "
                        + str(obj.velocities[-1][1] * scale)
                    )
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


def is_detection_outdated_or_not_confirmed(obj, detector):
    return (
        detector.frame_number - obj.frames_observed[-1] > detector.conf["no_more_show_after_x_frames"]
        or obj.detection_is_confirmed is False
    )
