import motmetrics as mm
import numpy as np
import pandas as pd


def prepare_data_for_mot_metrics(ground_truth_source, test_source):
    df_tsource = pd.read_csv(test_source, delimiter=",")
    df_gt = pd.read_csv(ground_truth_source, delimiter=",")
    df_gt["id"] = df_gt["id"] - 245
    df_tsource.drop(
        df_tsource[df_tsource["classification"] != "fish"].index, inplace=True
    )
    df_tsource.drop(columns=["classification", "v_yr"], inplace=True)
    df_tsource["v_x"] = -1
    df_tsource["v_y"] = -1
    df_tsource["contour_area"] = -1
    df_tsource["v_xr"] = -1
    df_tsource.to_csv("test.csv", header=False, index=False)
    test_source = "test.csv"
    df_gt.to_csv("ground_truth.csv", header=False, index=False)
    ground_truth_source = "ground_truth.csv"
    return ground_truth_source, test_source


def motMetricsEnhancedCalculator(ground_truth_source, test_source):

    ground_truth = np.loadtxt(ground_truth_source, delimiter=",", encoding="utf-8-sig")

    # load tracking output
    test = np.loadtxt(test_source, delimiter=",", encoding="utf-8-sig")

    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)

    # Max frame number maybe different for gt and t files
    for frame in range(int(ground_truth[:, 1].max())):
        frame += 1  # detection and frame numbers begin at 1
        # print(frame)
        # select id, x, y, width, height for current frame
        # required format for distance calculation is X, Y, Width, Height \
        # We already have this format
        if frame == 646:
            print("debug")
        ground_truth_detections = ground_truth[
            ground_truth[:, 1] == frame, 1:6
        ]  # select all detections in gt
        test_detections = test[test[:, 1] == frame, 1:6]  # select all detections in t

        C = mm.distances.iou_matrix(
            ground_truth_detections[:, 1:], test_detections[:, 1:], max_iou=0.5
        )  # format: gt, t

        # Call update once for per frame.
        # format: gt object ids, t object ids, distance
        acc.update(
            ground_truth_detections[:, 0].astype("int").tolist(),
            test_detections[:, 0].astype("int").tolist(),
            C,
        )

    mh = mm.metrics.create()

    summary = mh.compute(
        acc,
        metrics=[
            "num_frames",
            "idf1",
            "idp",
            "idr",
            "recall",
            "precision",
            "num_objects",
            "mostly_tracked",
            "partially_tracked",
            "mostly_lost",
            "num_false_positives",
            "num_misses",
            "num_switches",
            "num_fragmentations",
            "mota",
            "motp",
        ],
        name="acc",
    )

    strsummary = mm.io.render_summary(
        summary,
        # formatters={'mota' : '{:.2%}'.format},
        namemap={
            "idf1": "IDF1",
            "idp": "IDP",
            "idr": "IDR",
            "recall": "Rcll",
            "precision": "Prcn",
            "num_objects": "GT",
            "mostly_tracked": "MT",
            "partially_tracked": "PT",
            "mostly_lost": "ML",
            "num_false_positives": "FP",
            "num_misses": "FN",
            "num_switches": "IDsw",
            "num_fragmentations": "FM",
            "mota": "MOTA",
            "motp": "MOTP",
        },
    )
    print(strsummary)


if __name__ == "__main__":
    ground_truth_source = "labels/hand_labeled.csv"
    test_source = "labels/demo_sample_sonar_recording.csv"
    ground_truth_source, test_source = prepare_data_for_mot_metrics(
        ground_truth_source, test_source
    )
    motMetricsEnhancedCalculator(ground_truth_source, test_source)
