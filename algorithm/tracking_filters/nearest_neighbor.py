import numpy as np


def closest_point(point, points):
    points = np.asarray(points)
    dist_2 = np.sqrt(np.sum((points - point) ** 2, axis=1))
    min_index = np.argmin(dist_2)
    return min_index, dist_2[min_index]


def associate_detections(
    detections, object_history, frame_number, conf, max_association_distance_px
):
    if len(object_history) == 0:
        object_history = detections
        return object_history

    existing_object_midpoints = [
        existing_object.midpoints[-1]
        for _, existing_object in object_history.items()
        if (
            frame_number - existing_object.frames_observed[-1]
            < conf["phase_out_after_x_frames"]
        )
    ]

    if len(existing_object_midpoints) == 0:
        for _, detection in detections.items():
            object_history[detection.ID] = detection
        return object_history

    existing_object_ids = [
        key
        for key, existing_object in object_history.items()
        if (
            frame_number - existing_object.frames_observed[-1]
            < conf["phase_out_after_x_frames"]
        )
    ]

    new_objects = []
    associations = {}
    # Loop the detections
    for _, detection in detections.items():
        # Find the closest existing object
        min_id, min_dist = closest_point(
            detection.midpoints[-1], existing_object_midpoints
        )
        if min_dist < max_association_distance_px:
            if existing_object_ids[min_id] in associations.keys():
                if associations[existing_object_ids[min_id]]["distance"] > min_dist:
                    new_objects.append(
                        associations[existing_object_ids[min_id]]["detection"]
                    )
                    associations[existing_object_ids[min_id]] = {
                        "detection_id": detection.ID,
                        "distance": min_dist,
                        "detection": detection,
                    }
                new_objects.append(detection)
            else:
                associations[existing_object_ids[min_id]] = {
                    "detection_id": detection.ID,
                    "distance": min_dist,
                    "detection": detection,
                }
        else:
            new_objects.append(detection)

    for new_object in new_objects:
        object_history[new_object.ID] = new_object

    for existing_object_id, associated_detection in associations.items():
        object_history[existing_object_id].update_object(
            detections[associated_detection["detection_id"]]
        )

    return object_history
