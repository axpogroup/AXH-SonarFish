import argparse
from pathlib import Path

from sklearn.linear_model import LogisticRegression

from analysis.classification_utils.features import FeatureGenerator


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
    gen.calc_feature_dfs()

    gen.do_binary_classification(
        LogisticRegression(class_weight="balanced"),
        kfold_n_splits=9,
        features=["median_smoothed_curvature_30", "average_bbox_size"],
        manual_noise_thresholds=[
            ("average_distance_from_start/traversed_distance", "smaller", 0.08),
        ],
        features_flow_area=[],
        manual_noise_thresholds_flow_area=[
            ("median_curvature", "larger", 0.40),
            ("average_distance_from_start/traversed_distance", "smaller", 0.10),
            ("x_avg", "smaller", 60.0),
            ("average_overlap_ratio", "smaller", 1.5),
        ],
        class_overrides_flow_area=[
            ("median_curvature", "smaller", 0.05, 2),
        ],
    )

    print("Calculating train/val performance metrics: ")
    gen.calculate_metrics(
        beta_vals=[2, 3],
        make_plots=False,
        verbosity=1,
        distinguish_flow_areas=True,
    )

    print("Plotting track pairings...")
    gen.plot_track_pairings(
        mask_to_show="flow_area_mask",
        metric_to_show="median_curvature",
        show_track_id=True,
        plot_test_data=True,
        save_dir=args.job_output_dir,
        n_labels=3,
        plot_results_individually=True,
    )
    print("Saving classified tracks to csv...")
    gen.save_classified_tracks_to_csv(save_dir=args.job_output_dir)

    # TODO: figure out why this fails here but works in the notebook


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
