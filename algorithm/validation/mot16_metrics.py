import argparse
from pathlib import Path

import motmetrics as mm
import pandas as pd
import yaml
from numpy import ndarray


def prepare_data_for_mot_metrics(
    ground_truth_source: Path, test_source: Path
) -> tuple[ndarray, ndarray]:
    df_tsource = pd.read_csv(test_source, delimiter=",")
    df_gt = pd.read_csv(ground_truth_source, delimiter=",")
    df_tsource.drop(
        df_tsource[df_tsource["classification"] != "fish"].index, inplace=True
    )
    df_tsource.drop(columns=["classification", "v_yr"], inplace=True)
    df_tsource["v_x"] = -1
    df_tsource["v_y"] = -1
    df_tsource["contour_area"] = -1
    df_tsource["v_xr"] = -1
    return df_gt.to_numpy(), df_tsource.to_numpy()


def mot_metrics_enhanced_calculator(ground_truth, test):

    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)

    # Max frame number maybe different for gt and t files
    for frame in range(int(ground_truth[:, 0].max())):
        frame += 1  # detection and frame numbers begin at 1
        # print(frame)
        # select id, x, y, width, height for current frame
        # required format for distance calculation is X, Y, Width, Height \
        # We already have this format
        ground_truth_detections = ground_truth[
            ground_truth[:, 0] == frame, 1:6
        ]  # select all detections in gt
        test_detections = test[test[:, 0] == frame, 1:6]  # select all detections in t

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
    return summary.to_dict("records")[0]


if __name__ == "__main__":
    argParser = argparse.ArgumentParser(
        description="Run the fish detection algorithm with a settings .yaml file."
    )
    argParser.add_argument(
        "-yf", "--yaml_file", help="path to the YAML settings file", required=True
    )
    argParser.add_argument("-if", "--input_file", help="path to the input video file")

    args = argParser.parse_args()

    with open(args.yaml_file) as f:
        settings_dict = yaml.load(f, Loader=yaml.SafeLoader)
        if args.input_file is not None:
            print("replacing input file.")
            settings_dict["file_name"] = args.input_file
    file_name_prefix = Path(settings_dict["file_name"]).stem
    ground_truth_source = Path(settings_dict["ground_truth_directory"]) / Path(
        file_name_prefix + "_ground_truth.csv"
    )
    test_source = (
        Path(settings_dict["test_directory"])
        / file_name_prefix
        / Path(file_name_prefix + ".csv")
    )

    ground_truth_source, test_source = prepare_data_for_mot_metrics(
        ground_truth_source, test_source
    )
    mot_metrics_enhanced_calculator(ground_truth_source, test_source)
