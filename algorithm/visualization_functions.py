import copy

import cv2 as cv
import numpy as np

FIRST_ROW = [
    "gray_boosted",
    "short_mean",
    "long_mean",
    "difference",
    "absolute_difference",
]
SECOND_ROW = ["difference_thresholded", "median_filter", "binary", "dilated", "blank"]


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
                    debug=False,
                    classifications=False,
                ),
                retrieve_frame(None, processed_frame),
                retrieve_frame(None, processed_frame),
                draw_detector_output(
                    detector,
                    retrieve_frame("dilated", processed_frame, puttext="detections"),
                    debug=True,
                ),
                retrieve_frame(None, processed_frame),
            ),
            axis=1,
        )
        disp = np.concatenate((first_row_images, second_row_images, third_row_images))

    else:
        disp = draw_detector_output(
            detector,
            retrieve_frame("raw", processed_frame),
            classifications=True,
            runtiming=True,
            fullres=True,
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
    debug=False,
    runtiming=False,
    fullres=False,
    only_runtime=False,
):
    output = img
    if not only_runtime:
        output = draw_objects(
            detector,
            output,
            classifications=classifications,
            debug=debug,
            fullres=fullres,
        )
    # TOD Currently broken.
    # if runtiming:
    #     cv.rectangle(output, (1390, 25), (1850, 155), (0, 0, 0), -1)
    #     color = (255, 255, 255)
    #     cv.putText(
    #         output,
    #         f"Frame no. {detector.frame_number}",
    #         (1500, 50),
    #         cv.FONT_HERSHEY_SIMPLEX,
    #         0.75,
    #         color,
    #         2,
    #     )
    #     cv.putText(
    #         output,
    #         f"{detector.enhance_time_ms} ms - Enhancement",
    #         (1400, 80),
    #         cv.FONT_HERSHEY_SIMPLEX,
    #         0.75,
    #         color,
    #         2,
    #     )
    #     cv.putText(
    #         output,
    #         f"{detector.detection_tracking_time_ms} ms - Detection & Tracking",
    #         (1400, 110),
    #         cv.FONT_HERSHEY_SIMPLEX,
    #         0.75,
    #         color,
    #         2,
    #     )
    #     if detector.total_runtime_ms > 40:
    #         color = (100, 100, 255)
    #     if detector.total_runtime_ms == 0:
    #         detector.total_runtime_ms = 1
    #     cv.putText(
    #         output,
    #         f"{detector.total_runtime_ms} ms - Total - FPS: {int(1000/detector.total_runtime_ms)}",
    #         (1400, 140),
    #         cv.FONT_HERSHEY_SIMPLEX,
    #         0.75,
    #         color,
    #         2,
    #     )
    return output


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


def draw_objects(detector, img, debug=False, classifications=False, fullres=False):
    for ID, obj in detector.current_objects.items():
        if obj.show[-1]:
            if classifications:
                if fullres:
                    draw_object_classifications_box(obj, img, detector.downsample)
                else:
                    draw_object_classifications_box(obj, img)
            else:
                draw_object_bounding_box(obj, img, color=(255, 0, 0))
                draw_object_past_midpoints(obj, img, color=(255, 0, 0))
        elif (obj.frames_observed[-1] == detector.frame_number) & debug:
            draw_object_bounding_box(obj, img, color=(20, 20, 20))
            draw_object_past_midpoints(obj, img, color=(20, 20, 20))

        elif debug:
            draw_object_bounding_box(obj, img, color=(50, 50, 50))

        # if debug:
        # cv.circle(img, (obj.midpoints[-1][0], obj.midpoints[-1][1]),
        #           int(detector.max_association_dist/2), (0, 0, 255), 1)

    return img


def draw_object_bounding_box(object, img, color):
    x, y, w, h = cv.boundingRect(object.contours[-1])
    cv.rectangle(img, (x, y), (x + w, y + h), color, 1)
    # cv.putText(
    #     img,
    #     (object.classification[-1] + " " + str(self.ID)),
    #     (x, y - 10),
    #     cv.FONT_HERSHEY_SIMPLEX,
    #     0.75,
    #     color,
    #     2,
    # )
    # if object.median_v is not None:
    #     cv.putText(img, f"v_x, v_y: {object.median_v[0]}, {object.median_v[1]}",
    #                (x, y + 10), cv.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    return img


def draw_object_classifications_box(object, img, downsampled=100):
    x, y, w, h = cv.boundingRect(object.contours[-1])
    upsample_factor = int(100 / downsampled)
    x, y, w, h = (
        x * upsample_factor,
        y * upsample_factor,
        w * upsample_factor,
        h * upsample_factor,
    )
    if object.classifications[-1] == "Fisch":
        color = (0, 255, 0)
    else:
        color = (250, 150, 150)

    cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
    if downsampled != 100:
        cv.putText(
            img,
            (object.classifications[-1]),
            (x, y - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
        )
        # if object.median_v is not None:
        #     str_1 = (
        #         "{:.2f}".format(object.velocity[-1][0] * 4)
        #         + ", "
        #         + "{:.2f}".format(object.velocity[-1][1] * 4)
        #     )
        #     cv.putText(
        #         img,
        #         f"v_x, v_y: {str_1}",
        #         (x, y - 10),
        #         cv.FONT_HERSHEY_SIMPLEX,
        #         0.75,
        #         color,
        #         2,
        #     )

    return img


def draw_object_past_midpoints(object, img, color):
    for point in object.midpoints:
        cv.circle(img, point, 2, color, -1)
    return img
