import argparse
import yaml
from pathlib import Path
from copy import deepcopy

import pandas as pd
import numpy as np

from analysis.classification_utils.classifier_evaluation import (
    ProbaLogisticRegression,
    ProbaXGBClassifier,
    train_and_evaluate_model,
    predict,
)
from analysis.classification_utils.features import FeatureGenerator, TrackPlotter
from analysis.classification_utils.metrics import get_confusion_matrix
from analysis.classification_utils.dataframe_manipulations import save_classified_trajectories

random_state = 3
features_to_use = [
    "v_50th_percentile",
    "average_distance_from_start",
    "v_95th_percentile",
    "contour_area",
    "average_distance_from_start/traversed_distance",
]
with open("config.yaml", "r") as f:
    settings = yaml.safe_load(f)
    if settings.classifier == "LogisticRegression":
        classifier = ProbaLogisticRegression
    elif settings.classifier == "XGBClassifier":
        classifier = ProbaXGBClassifier


def parse_arguments():
    parser = argparse.ArgumentParser(description="Classify the Tracked Objects")
    parser.add_argument("--train_val_data_dir", required=True)
    parser.add_argument("--train_val_gt_data_dir", required=True)
    parser.add_argument("--test_data_dir", required=True)
    parser.add_argument("--job_output_dir", required=True)
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()
    return args


def main(args):
    train_val_data_files = sorted(list(Path(args.train_val_data_dir).rglob("*.csv")))
    train_val_gt_data_files = sorted(list(Path(args.train_val_gt_data_dir).rglob("*.csv")))
    test_data_files = sorted(list(Path(args.test_data_dir).rglob("*.csv")))
    print(f"Found {train_val_data_files} training data files.")
    print(f"Found {train_val_gt_data_files} training ground truth data files.")
    print(f"Found {test_data_files} test data files.")
    gen = FeatureGenerator(
        measurements_csv_paths=train_val_data_files,
        gt_csv_paths=train_val_gt_data_files,
        test_csv_paths=test_data_files,
        test_gt_csv_paths=[],
        min_track_length=30,
        force_feature_recalc=False,
        force_test_feature_recalc=True,
        min_overlapping_ratio=0.5,
        load_old_measurements=False,
        load_old_test_files=False,
        rake_mask_path="./masks/stroppel_rake_front_mask.png",
        flow_area_mask_path="./masks/stroppel_flow_area_mask.png",
        non_flow_area_mask_path="./masks/stroppel_non_flow_area_mask.png",
        trajectory_min_overlap_ratio=0.15,
    )
    gen = FeatureGenerator(
        gt_fish_id_yaml=train_val_data_yaml,
        measurements_csv_dir=measurements_csv_dir,
        test_csv_paths=test_csv_paths,
        min_track_length=20,
        force_feature_recalc=True,
        force_test_feature_recalc=True,
        min_overlapping_ratio=0.5,
        load_old_measurements=False,
        load_old_test_files=False,
        rake_mask_path="../demo/masks/stroppel_rake_front_mask.png",
        flow_area_mask_path="../demo/masks/sonar_controls.png",
        trajectory_min_overlap_ratio=0.15,
    )
    gen.calc_feature_dfs()

    test_plotter = TrackPlotter(deepcopy(gen.test_dfs), gen.masks)
    plotter = TrackPlotter(deepcopy(gen.measurements_dfs), gen.masks)
    feature_df = pd.concat(deepcopy(plotter.measurements_dfs)).groupby("id").first().select_dtypes(include=[np.number])
    imbalance = feature_df["gt_label"].value_counts()[0] / sum(feature_df["gt_label"].value_counts())

    print("Training classifier...")
    metrics, y_pred, trained_classifier, trained_scaler = train_and_evaluate_model(
        feature_df,
        classifier(
            proba_threshold=settings.proba_threshold,
            random_state=settings.random_state,
            scale_pos_weight=imbalance,
            class_weight="balanced",
            verbosity=0,
        ),
        metrics_to_show=["Accuracy", "Precision", "Recall", "F1_score", "F2_score"],
        features=features_to_use,
    )
    confusion_matrix = get_confusion_matrix(feature_df["gt_label"], y_pred)
    print("Metrics: ", metrics)
    print("Confusion matrix: ", confusion_matrix)

    print("Calculating train/val performance metrics: ")
    test_feature_df = (
        pd.concat(deepcopy(test_plotter.measurements_dfs)).groupby("id").first().select_dtypes(include=[np.number])
    )

    y_test, y_test_proba = predict(
        test_feature_df,
        trained_classifier,
        trained_scaler,
        features_to_use,
    )
    data = {
        "classification_v2": y_test,
        "xgboost_proba": y_test_proba[:, 1],
    }
    df = pd.DataFrame(data, index=test_feature_df.index)
    test_plotter.overwrite_classification_v2(df)

    print("Plotting track pairings...")
    test_plotter.plot_track_pairings(
        show_track_id=True,
        mask_to_show="flow_area_mask",
        column_with_label="classification_v2",
        figsize=(20, 20),
        save_dir=args.job_output_dir,
        n_labels=3,
        plot_results_individually=True,
    )

    print("Saving classified tracks to csv...")
    save_classified_trajectories(
        test_plotter.measurements_dfs,
        test_data_files,
        save_dir=args.job_output_dir,
        name_extension=f"_classification_min_track_length_{settings.min_track_length}",
    )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
