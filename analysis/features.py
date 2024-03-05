import json
from pathlib import Path
from typing import Callable, Iterator, Optional, Union

import cv2 as cv
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from motmetrics.distances import boxiou
from scipy.optimize import linear_sum_assignment
from sklearn import metrics, preprocessing
from sklearn.model_selection import KFold
from tqdm import tqdm

from analysis.features_utils import calculate_features


def calculate_track_distance(
    measurements_track: pd.DataFrame,
    gt_track: pd.DataFrame,
    min_iou_thresh: float,
    min_overlap_ratio: float,
) -> float:
    if set(measurements_track["frame"]).isdisjoint(set(gt_track["frame"])):
        measurements_to_gt_dist = 1.0
    else:
        joined = measurements_track.set_index("frame").join(
            gt_track.set_index("frame"), how="outer", lsuffix="_measurements", rsuffix="_gt"
        )
        iou = boxiou(
            joined[["x_measurements", "y_measurements", "w_measurements", "h_measurements"]].values,
            joined[["x_gt", "y_gt", "w_gt", "h_gt"]].values,
        )
        iou[iou < min_iou_thresh] = np.nan
        iou_cost = 1.0 - iou
        if np.isnan(iou_cost).all():
            measurements_to_gt_dist = 1.0
        elif np.count_nonzero(~np.isnan(iou_cost)) / len(measurements_track) < min_overlap_ratio:
            measurements_to_gt_dist = 1.0
        else:
            measurements_to_gt_dist = np.nanmean(iou_cost)
    return measurements_to_gt_dist


def trace_window_metrics(group: pd.DataFrame) -> pd.Series:
    # Calculate the Euclidean distance between previous and current positions
    x_diff = np.diff(group["x"])
    y_diff = np.diff(group["y"])
    euclidean_distances = np.sqrt(x_diff**2 + y_diff**2)

    # Calculate the sum of distances over the group
    traversed_distance = np.sum(euclidean_distances)

    # Calculate the frame number difference
    frame_diff = group["frame"].iloc[-1] - group["frame"].iloc[0]

    return pd.Series({"traversed_distance": traversed_distance, "frame_diff": frame_diff})


def load_csv_with_tiles(path: Path) -> pd.DataFrame:
    csv_with_tiles_df = pd.read_csv(path, delimiter=",")
    csv_with_tiles_df["raw_image_tile"] = csv_with_tiles_df["raw_image_tile"].apply(lambda x: np.array(json.loads(x)))
    csv_with_tiles_df["image_tile"] = csv_with_tiles_df["image_tile"].apply(lambda x: np.array(json.loads(x)))
    return csv_with_tiles_df


def filter_features(measurements_df, min_overlapping_ratio):
    measurements_df = measurements_df[measurements_df["average_overlap_ratio"] > min_overlapping_ratio]
    return measurements_df


class FeatureGenerator(object):
    def __init__(
        self,
        measurements_csv_paths: list[Union[str, Path]],
        gt_csv_paths: list[Union[str, Path]],
        rake_mask_path: Union[str, Path],
        flow_area_mask_path: Union[str, Path],
        non_flow_area_mask_path: Union[str, Path],
        min_track_length: int = 10,
        force_feature_recalc: bool = False,
        min_overlapping_ratio: int = 1,
        trajectory_min_iou_thresh: float = 0.4,
        trajectory_min_overlap_ratio: float = 0.3,
        trajectory_max_iou_track_distance: float = 0.6,
    ):
        self.min_overlapping_ratio = min_overlapping_ratio
        self.min_track_length = min_track_length
        measurements_csv_paths = [Path(p) for p in measurements_csv_paths]
        gt_csv_paths = [Path(p) for p in gt_csv_paths]
        self.masks = {
            "rake_mask": self._read_mask(rake_mask_path),
            "flow_area_mask": self._read_mask(flow_area_mask_path),
            "non_flow_area_mask": self._read_mask(non_flow_area_mask_path),
        }
        self._calc_feature_dfs(measurements_csv_paths, gt_csv_paths, force_feature_recalc)
        self._map_measurements2gt_trajectory(
            trajectory_min_iou_thresh, trajectory_min_overlap_ratio, trajectory_max_iou_track_distance
        )

    def _calc_feature_dfs(
        self,
        measurements_csv_paths: list[Path],
        gt_csv_paths: list[Path],
        force_feature_recalc: bool,
    ) -> None:
        self.measurements_dfs, self.gt_dfs = [], []
        for idx, (measurements_df, gt_df) in enumerate(
            self._read_csvs(measurements_csv_paths, gt_csv_paths, force_feature_recalc)
        ):
            self.gt_dfs.append(gt_df)
            measurements_df["video_id"] = idx
            self.measurements_dfs.append(measurements_df)

    def _read_csvs(
        self,
        measurements_csv_paths: list[Path],
        gt_csv_paths: list[Path],
        force_feature_recalc: bool = False,
    ) -> Iterator[dict[str, Union[pd.DataFrame]]]:
        print("Calculating/reading features")
        for path in tqdm(measurements_csv_paths):
            cache_path = path.with_stem(
                path.stem + f"_cached_features_min_track_length_{self.min_track_length}"
            ).with_suffix(".csv")
            gt_df = pd.read_csv(gt_csv_paths[measurements_csv_paths.index(path)], delimiter=",")
            if cache_path.exists() and not force_feature_recalc:
                print(f"Reading cached features from {cache_path}")
                measurements_df = load_csv_with_tiles(cache_path)
                yield measurements_df, gt_df
            else:
                measurements_df = load_csv_with_tiles(path)
                value_counts_model = (
                    pd.DataFrame(measurements_df.id.value_counts())
                    .reset_index()
                    .rename(columns={"index": "id", "id": "occurences"})
                )
                measurements_df = measurements_df[
                    measurements_df["id"].isin(
                        value_counts_model[value_counts_model.occurences >= self.min_track_length]["id"]
                    )
                ]
                measurements_df = calculate_features(measurements_df, self.masks)
                measurements_df = filter_features(measurements_df, self.min_overlapping_ratio)
                save_df = measurements_df.copy()
                save_df["image_tile"] = save_df["image_tile"].apply(lambda x: x.tolist())
                save_df["raw_image_tile"] = save_df["raw_image_tile"].apply(lambda x: x.tolist())
                save_df.to_csv(cache_path, index=False)
                yield measurements_df, gt_df

    @staticmethod
    def _read_mask(mask_path: Union[str, Path]) -> np.ndarray:
        mask = cv.imread(Path(mask_path).as_posix(), cv.IMREAD_GRAYSCALE)
        assert mask is not None, f"Could not read mask from {mask_path}"
        mask = mask[49:1001, 92:1831]  # Drop the border
        return cv.resize(mask, (480, 270)) > 0

    def _map_measurements2gt_trajectory(
        self, min_iou_thresh: float = 0.4, min_overlap_ratio: float = 0.3, max_iou_track_distance: float = 0.6
    ) -> None:
        print("Mapping measurements to ground truth trajectories")

        all_measurements_gt_pairs, all_measurements_gt_pairs_secondary = [], []
        for df_idx, (measurements_df, gt_df) in enumerate(zip(self.measurements_dfs, self.gt_dfs)):
            model_detections = measurements_df
            ground_truth = gt_df

            track_distances = np.empty((len(model_detections["id"].unique()), len(ground_truth["id"].unique())))

            for measurements_idx, track_id in tqdm(enumerate(model_detections["id"].unique())):
                measurements_track = model_detections.loc[
                    model_detections["id"] == track_id, ["frame", "x", "y", "w", "h"]
                ]

                for gt_idx, gt_id in enumerate(ground_truth["id"].unique()):
                    gt_track = ground_truth.loc[ground_truth["id"] == gt_id, ["frame", "x", "y", "w", "h"]]
                    track_distances[measurements_idx, gt_idx] = calculate_track_distance(
                        measurements_track, gt_track, min_iou_thresh, min_overlap_ratio
                    )

            measurements_gt_pairs, dist_matrix_indices = self._calculate_trajectory_pairs(
                track_distances, model_detections, ground_truth, max_iou_track_distance
            )

            track_distances_copy = track_distances.copy()
            track_distances_copy[dist_matrix_indices] = 1.0
            measurements_gt_pairs_secondary, _ = self._calculate_trajectory_pairs(
                track_distances_copy, model_detections, ground_truth, max_iou_track_distance
            )

            self.measurements_dfs[df_idx]["gt_label"] = "noise"
            self.measurements_dfs[df_idx].loc[
                measurements_df["id"].isin(
                    np.hstack((measurements_gt_pairs[:, 0], measurements_gt_pairs_secondary[:, 0]))
                ),
                "gt_label",
            ] = "fish"

            all_measurements_gt_pairs.extend(measurements_gt_pairs)
            all_measurements_gt_pairs_secondary.extend(measurements_gt_pairs_secondary)

        self.all_measurements_gt_pairs = dict(all_measurements_gt_pairs)
        self.all_measurements_gt_pairs_secondary = dict(all_measurements_gt_pairs_secondary)

    def _calculate_trajectory_pairs(
        self,
        track_distances: np.ndarray,
        model_detections: pd.DataFrame,
        ground_truth: pd.DataFrame,
        max_iou_track_distance: float,
    ) -> tuple[np.array, np.array]:

        row_indices, col_indices = linear_sum_assignment(track_distances)
        dist_matrix_indices = np.where(track_distances[row_indices, col_indices] < max_iou_track_distance)
        dist_matrix_row_indices = np.array(row_indices)[dist_matrix_indices]
        dist_matrix_col_indices = np.array(col_indices)[dist_matrix_indices]
        measurements_gt_pairs = np.vstack(
            (model_detections.id.unique()[dist_matrix_row_indices], ground_truth.id.unique()[dist_matrix_col_indices])
        ).T

        return measurements_gt_pairs, (dist_matrix_row_indices, dist_matrix_col_indices)

    def plot_track_pairings(
        self,
        metric_to_show: Optional[str] = None,
        mask_to_show: Optional[str] = None,
    ) -> None:
        if "assigned_label" not in self.measurements_dfs[0].columns:
            raise ValueError(
                "No clustering has been performed yet. Please perform clustering with the do_clustering method."
            )

        model_detections = pd.concat(self.measurements_dfs)
        ground_truth = pd.concat(self.gt_dfs)

        n_labels = model_detections["assigned_label"].nunique()
        show_legend = True if n_labels <= 6 else False
        colormap = cm.get_cmap("viridis", n_labels + 1)  # +1 for the ground truth color

        _, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(10, 10))
        if mask_to_show:
            ax.imshow(self.masks[mask_to_show], cmap="gray", alpha=0.2)
        plt.gca().invert_yaxis()

        # Plot assigned tracks
        for measurements_track_id, gt_track_id in self.all_measurements_gt_pairs.items():
            self.plot_tracks_and_annotations(ax, colormap, measurements_track_id, model_detections, metric_to_show)
            gt_track_df = ground_truth[ground_truth.id == gt_track_id]
            ax.plot(gt_track_df.x, gt_track_df.y, alpha=0.5, color=colormap(0), linestyle="dashed")

        for measurements_track_id, gt_track_id in self.all_measurements_gt_pairs_secondary.items():
            self.plot_tracks_and_annotations(ax, colormap, measurements_track_id, model_detections, metric_to_show)

        ax.set(ylabel="y", title="matched trajectories with assigned label", ylim=[270, 0], xlim=[0, 480])
        ax.set_aspect("equal", adjustable="box")

        legend_elements = [Line2D([0], [0], color=colormap(0), lw=2, label="manually labeled")]
        if show_legend:
            for i in range(n_labels):
                legend_elements.append(Line2D([0], [0], color=colormap(i + 1), lw=2, label=f"assigned label {i}"))
        ax.legend(handles=legend_elements)

        plt.show()

        # Plot tracks that were not assigned to a ground truth
        _, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(10, 10))
        if mask_to_show:
            ax.imshow(self.masks[mask_to_show], cmap="gray", alpha=0.2)
        plt.gca().invert_yaxis()
        plt.gca().invert_yaxis()
        for measurements_track_id in model_detections.id.unique():
            if (measurements_track_id not in self.all_measurements_gt_pairs.keys()) and (
                measurements_track_id not in self.all_measurements_gt_pairs_secondary.keys()
            ):
                self.plot_tracks_and_annotations(ax, colormap, measurements_track_id, model_detections, metric_to_show)
        ax.set(ylabel="y", title="non-matched trajectories with assigned label", ylim=[270, 0], xlim=[0, 480])
        ax.set_aspect("equal", adjustable="box")

        legend_elements = [Line2D([0], [0], color=colormap(0), lw=2, label="Label")]
        if show_legend:
            for i in range(n_labels):
                legend_elements.append(Line2D([0], [0], color=colormap(i + 1), lw=2, label=f"label {i}"))
        ax.legend(handles=legend_elements)

        plt.show()

    def plot_tracks_and_annotations(
        self,
        ax,
        colormap,
        measurements_track_id,
        model_detections,
        metric_to_show: Optional[str],
    ) -> None:
        track_df = model_detections[model_detections.id == measurements_track_id]
        ax.plot(
            track_df.x,
            track_df.y,
            color=colormap(int(track_df.assigned_label.iloc[0]) + 1),
        )
        metric_annotation = str(track_df[metric_to_show].iloc[0])[:4] if metric_to_show else ""
        ax.annotate(
            f"{measurements_track_id}, {metric_annotation}",
            (track_df.x.iloc[0], track_df.y.iloc[0]),
        )

    def plot_image_tiles_along_trajectory(
        self,
        measurements_track_id: int,
        every_nth_frame: int,
        raw: bool = False,
    ) -> None:
        feature_name = "raw_image_tile" if raw else "image_tile"
        track_df = self._get_track_df_by_id(measurements_track_id)

        for idx, row in track_df.iterrows():
            if idx % every_nth_frame == 0:
                image_tile = np.squeeze(row[feature_name])
                if raw:
                    plt.imshow(image_tile[:, :, ::-1])
                else:
                    plt.imshow(image_tile, cmap="gray")
                plt.title(f"Frame {row.frame}")
                plt.show()

    def show_trajectory_numeric_features(
        self,
        measurements_track_id: int,
        boxplot_split_thresholds: list[float] = [1],
    ) -> None:
        boxplot_split_thresholds = sorted(boxplot_split_thresholds)
        features_to_print = [
            feature for feature in self.feature_names if feature not in ["image_tile", "raw_image_tile", "video_id"]
        ]
        features_to_plot = [
            feature for feature in features_to_print if feature not in ["classification", "gt_label", "assigned_label"]
        ]

        track_df = self._get_track_df_by_id(measurements_track_id).iloc[0]
        all_tracks_df = self.stacked_dfs.groupby(["video_id", "id"]).first().reset_index()

        # Calculate medians
        medians = all_tracks_df[features_to_plot].median()

        # Split features based on thresholds
        subplots = len(boxplot_split_thresholds) + 1
        _, axs = plt.subplots(subplots)

        for i, threshold in enumerate(boxplot_split_thresholds):
            if i == 0:
                features_above_threshold = [feature for feature in features_to_plot if medians[feature] <= threshold]
            elif i < len(boxplot_split_thresholds) - 1:
                features_above_threshold = []
                for feature in features_to_plot:
                    if (
                        medians[feature] > boxplot_split_thresholds[i - 1]
                        and medians[feature] <= boxplot_split_thresholds[i]
                    ):
                        features_above_threshold.append(feature)
            else:
                features_above_threshold = [feature for feature in features_to_plot if medians[feature] > threshold]
            if not features_above_threshold:
                continue
            axs[i].boxplot(all_tracks_df[features_above_threshold].values, labels=features_above_threshold)
            axs[i].plot(
                range(1, len(features_above_threshold) + 1), track_df[features_above_threshold].values.tolist(), "ro"
            )
            axs[i].set_title(f"Numeric Features Distribution (Median > {threshold})")

        features_below_threshold = [
            feature for feature in features_to_plot if medians[feature] <= boxplot_split_thresholds[-1]
        ]
        axs[-1].boxplot(all_tracks_df[features_below_threshold].values, labels=features_below_threshold)
        axs[-1].plot(
            range(1, len(features_below_threshold) + 1), track_df[features_below_threshold].values.tolist(), "ro"
        )
        axs[-1].set_title(f"Numeric Features Distribution (Median <= {boxplot_split_thresholds[-1]})")

        for ax in axs:
            ax.set_xlabel("Features")
            ax.set_ylabel("Values")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        plt.subplots_adjust(hspace=1.5)  # Adjust the value as needed
        plt.show()

        max_name_length = max([len(feat) for feat in features_to_print])
        for feat in features_to_print:
            print(f"{feat:<{max_name_length}} {track_df[feat]}")

    def _get_track_df_by_id(self, track_id: int) -> pd.DataFrame:
        track_df = self.stacked_dfs[self.stacked_dfs.id == track_id]
        assert track_df["video_id"].nunique() == 1, "The track is not unique to a video"
        return track_df

    def do_clustering(self, features: list[str], clustering_method: Callable, n_clusters: int):
        X = pd.concat([df[features] for df in self.measurements_dfs])
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)
        clustering = clustering_method(n_clusters=n_clusters)
        clustering.fit(X)
        for idx, df in enumerate(self.measurements_dfs):
            X_val = scaler.transform(df[features])
            self.measurements_dfs[idx]["assigned_label"] = clustering.predict(X_val)

    def do_binary_classification(
        self,
        model: Callable,
        features: list[str],
        kfold_n_splits: int,
        distinguish_flow_areas: bool = False,
    ):
        if distinguish_flow_areas:
            df = self.stacked_dfs.groupby(["video_id", "id"]).first().reset_index()
            flow_area_indices = df["flow_area_time_ratio"] > 0.5
            df.loc[flow_area_indices, "assigned_label"] = self._do_binary_classification_for_trajectory_subset(
                df[flow_area_indices], model, features, kfold_n_splits
            )
            df.loc[~flow_area_indices, "assigned_label"] = self._do_binary_classification_for_trajectory_subset(
                df[~flow_area_indices], model, features, kfold_n_splits
            )
        else:
            df = self.stacked_dfs.groupby(["video_id", "id"]).first().reset_index()
            df["assigned_label"] = self._do_binary_classification_for_trajectory_subset(
                df, model, features, kfold_n_splits
            )

        for idx, measurement_df in enumerate(self.measurements_dfs):
            measurement_df.drop(columns=["assigned_label"], inplace=True)
            video_id = measurement_df["video_id"].iloc[0]
            right_df = df[df["video_id"] == video_id][["id", "assigned_label"]]
            self.measurements_dfs[idx] = measurement_df.merge(right_df, on="id", how="left")

    @staticmethod
    def _do_binary_classification_for_trajectory_subset(
        df: pd.DataFrame,
        model: Callable,
        features: list[str],
        kfold_n_splits: int,
    ) -> np.array:
        X = np.array(df[features])
        y = np.array([1 if lbl == "fish" else 0 for lbl in df["gt_label"]])

        kf = KFold(n_splits=kfold_n_splits)
        y_kfold = np.empty(y.shape)
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train, X_val = scaler.transform(X_train), scaler.transform(X_val)
            y_train, _ = y[train_index], y[val_index]

            model = model.fit(X_train, y_train)
            y_kfold[val_index] = model.predict(X_val)

        return y_kfold

    def do_thresholding_classification(
        self,
        feature_thresholding_values: Optional[dict[str, float]] = None,
    ) -> None:
        self.stacked_dfs["assigned_label"] = "fish"
        for feat, thresh in feature_thresholding_values.items():
            if feat.endswith("_min"):
                feat = feat.replace("_min", "")
                self.stacked_dfs.loc[self.stacked_dfs[feat] < thresh, "assigned_label"] = "noise"
            elif feat.endswith("_max"):
                self.stacked_dfs.loc[self.stacked_dfs[feat] > thresh, "assigned_label"] = "noise"
            else:
                print(f"Invalid feature threshold name: {feat}. Must end with '_min' or '_max'")

    def calculate_metrics(self, beta_vals: list[int] = [2], make_plots: bool = False):
        df = self.stacked_dfs.groupby("id").first().reset_index()
        inputs = [
            df["gt_label"].apply(lambda x: 1 if x == "fish" else 0),
            df["assigned_label"],
        ]
        confusion_matrix = metrics.confusion_matrix(*inputs, normalize="true", labels=[0, 1])
        precision = metrics.precision_score(*inputs, labels=[0, 1], average="binary")
        recall = metrics.recall_score(*inputs, labels=[0, 1], average="binary")
        f1 = metrics.f1_score(*inputs, labels=[0, 1], average="binary")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 score: {f1}")
        fbeta = []
        for beta in beta_vals:
            fbeta.append(metrics.fbeta_score(*inputs, beta=beta, labels=[0, 1], average="binary"))
            print(f"F{beta} score: {fbeta[-1]}")
        if make_plots:
            metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=["noise", "fish"]).plot()
        else:
            print(f"Confusion matrix: {confusion_matrix}")
        return confusion_matrix, precision, recall, f1, fbeta

    @property
    def feature_names(self):
        return self.measurements_dfs[0].columns.tolist()[9:]

    @property
    def stacked_dfs(self):
        return pd.concat(self.measurements_dfs)
