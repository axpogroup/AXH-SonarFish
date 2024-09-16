from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import sys

sys.path.append(".")
from analysis.classification_utils.classifier_evaluation import (
    predict,
    train_and_evaluate_model,
)
from analysis.classification_utils.dataframe_manipulations import (
    save_classified_trajectories,
)
from analysis.classification_utils.features import FeatureGenerator, TrackPlotter
from settings import args, classification_settings


def main(args):
    train_val_data_files = sorted(list(Path(args.train_val_gt_data_dir).rglob("*.csv")))
    train_val_gt_data_files = sorted(list(Path(args.train_val_gt_data_dir).rglob("*.csv")))
    test_data_files = sorted(list(Path(args.files_to_classify_dir).rglob("*.csv")))
    print(f"Found {train_val_data_files} training data files.")
    print(f"Found {train_val_gt_data_files} training ground truth data files.")
    print(f"Found {test_data_files} test data files.")
    print(f"Settings: {classification_settings}")

    gen = FeatureGenerator(
        gt_fish_id_yaml=Path(args.train_val_gt_data_dir) / args.train_val_data_yaml,
        measurements_csv_dir=args.train_val_gt_data_dir,
        test_csv_paths=list(Path(args.files_to_classify_dir).rglob("*.csv")),
        min_track_length=classification_settings.min_track_length,
        force_feature_recalc=True,
        force_test_feature_recalc=True,
        min_overlapping_ratio=0.5,
        load_old_measurements=False,
        load_old_test_files=False,
        rake_mask_path=Path(__file__).parent / "analysis/demo/masks/stroppel_rake_front_mask.png",
        flow_area_mask_path=Path(__file__).parent / "analysis/demo/masks/sonar_controls.png",
        trajectory_min_overlap_ratio=0.15,
    )
    gen.calc_feature_dfs()

    plotter = TrackPlotter(deepcopy(gen.measurements_dfs), gen.masks)
    feature_df = pd.concat(deepcopy(plotter.measurements_dfs)).groupby("id").first().select_dtypes(include=[np.number])
    imbalance = feature_df["gt_label"].value_counts()[0] / sum(feature_df["gt_label"].value_counts())

    print("Training classifier...")
    classifier = deepcopy(classification_settings.classifier)
    if classification_settings.classifier_name == "ProbaXGBClassifier":
        classifier.scale_pos_weight = imbalance  # We need to know the training data distribution to set this parameter

    metrics, _, trained_classifier, trained_scaler = train_and_evaluate_model(
        feature_df,
        classifier,
        metrics_to_show=["Accuracy", "Precision", "Recall", "F1_score", "F2_score"],
        features=classification_settings.features_to_use,
    )
    print("Metrics: ", metrics)

    print("Calculating train/val performance metrics... ")
    test_plotter = TrackPlotter(deepcopy(gen.test_dfs), gen.masks)
    test_feature_df = (
        pd.concat(deepcopy(test_plotter.measurements_dfs)).groupby("id").first().select_dtypes(include=[np.number])
    )
    y_test, y_test_proba = predict(
        test_feature_df,
        trained_classifier,
        trained_scaler,
        classification_settings.features_to_use,
    )
    data = {
        "classification_v2": y_test,
        f"{classification_settings.classifier_name}_proba": y_test_proba[:, 1],
    }
    df = pd.DataFrame(data, index=test_feature_df.index)
    test_plotter.overwrite_classification_v2(df)

    print("Plotting track pairings...")
    test_plotter.plot_track_pairings(
        show_track_id=True,
        mask_to_show="flow_area_mask",
        column_with_label="classification_v2",
        figsize=(20, 20),
        save_dir=args.classified_detections_dir,
        n_labels=3,
        plot_results_individually=True,
    )

    print("Saving classified tracks to csv...")
    save_classified_trajectories(
        test_plotter.measurements_dfs,
        test_data_files,
        save_dir=args.classified_detections_dir,
        name_extension=f"_classification_min_track_length_{classification_settings.min_track_length}",
    )


if __name__ == "__main__":
    main(args)
