import copy

import cv2 as cv
import numpy as np

FIRST_ROW = [
    "gray_boosted",
    "short_mean",
    "long_mean",
    "difference",
]
SECOND_ROW = ["difference_thresholded", "median_filter", "binary", "dilated"]


def get_visual_output(detector, processed_frame, extensive=False):
    if extensive:
        first_row_images = np.ndarray(shape=(270, 0, 3), dtype="uint8")
        second_row_images = np.ndarray(shape=(270, 0, 3), dtype="uint8")
        for frame_type in FIRST_ROW:
            first_row_images = np.concatenate(
                (
                    first_row_images,
                    retrieve_frame(frame_type, processed_frame, puttext=frame_type),
                ),
                axis=1,
            )

        for frame_type in SECOND_ROW:
            second_row_images = np.concatenate(
                (
                    second_row_images,
                    retrieve_frame(frame_type, processed_frame, puttext=frame_type),
                ),
                axis=1,
            )

        third_row_images = np.concatenate(
            (
                draw_detector_output(
                    detector,
                    retrieve_frame("raw_downsampled", processed_frame, puttext="Final"),
                ),
                retrieve_frame(
                    "internal_external", processed_frame, puttext="internal_external"
                ),
                draw_detector_output(
                    detector,
                    retrieve_frame("binary", processed_frame, puttext="detections"),
                    paths=True,
                    association_dist=True,
                ),
                retrieve_frame("closed", processed_frame, puttext="closed"),
            ),
            axis=1,
        )
        disp = np.concatenate((first_row_images, second_row_images, third_row_images))

    else:
        disp = draw_detector_output(
            detector,
            retrieve_frame("raw", processed_frame),
            paths=True,
            fullres=True,
            association_dist=True,
        )

    return disp


def retrieve_frame(frame, frame_dict, puttext=None):
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


def draw_detector_output(
    detector,
    img,
    classifications=False,
    paths=False,
    fullres=False,
    association_dist=False,
):
    for ID, obj in detector.current_objects.items():
        if (
            detector.frame_number - obj.frames_observed[-1]
            > detector.conf["no_more_show_after_x_frames"]
        ):
            continue

        # if len(obj.frames_observed) < 50:
        #     continue

        # if (
        #         obj.occurences_in_last_x(
        #             detector.frame_number, detector.conf["min_occurences_in_last_x_frames"][1]
        #         )
        #         <= detector.conf["min_occurences_in_last_x_frames"][0]
        # ):
        #     continue

        if fullres:
            scale = int(100 / detector.conf["downsample"])
        else:
            scale = 1
        x, y, w, h = cv.boundingRect(obj.contours[-1])
        x, y, w, h = (
            x * scale,
            y * scale,
            w * scale,
            h * scale,
        )

        if classifications:
            if obj.classifications[-1] == "Fisch":
                color = (0, 255, 0)
            else:
                color = (250, 150, 150)

            cv.rectangle(img, (x, y), (x + w, y + h), color, 1 * scale)
            if fullres:
                cv.putText(
                    img,
                    (obj.classifications[-1]),
                    (x, y - 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    color,
                    2,
                )
        else:
            color = (255, 0, 0)
            cv.rectangle(img, (x, y), (x + w, y + h), color, thickness=1 * scale)

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
