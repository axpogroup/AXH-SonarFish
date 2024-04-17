import itertools
import json
import pickle
from copy import deepcopy
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Union

import cv2 as cv
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from motmetrics.distances import boxiou
from pandas import DataFrame
from scipy.optimize import linear_sum_assignment
from sklearn import metrics, preprocessing
from sklearn.model_selection import KFold
from tqdm import tqdm

from analysis.features_utils import calculate_features, calculate_flow_area_time_ratio


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
    try:
        csv_with_tiles_df["raw_image_tile"] = csv_with_tiles_df["raw_image_tile"].apply(
            lambda x: np.array(json.loads(x))
        )
    except KeyError:
        pass
    csv_with_tiles_df["image_tile"] = csv_with_tiles_df["image_tile"].apply(lambda x: np.array(json.loads(x)))
    return csv_with_tiles_df


class FeatureGenerator(object):
    def __init__(
        self,
        load_old_measurements: bool,
        load_old_test_files: bool,
        measurements_csv_paths: list[Union[str, Path]],
        gt_csv_paths: list[Union[str, Path]],
        test_csv_paths: Optional[list[Union[str, Path]]],
        rake_mask_path: Union[str, Path],
        flow_area_mask_path: Union[str, Path],
        non_flow_area_mask_path: Union[str, Path],
        test_gt_csv_paths: Optional[list[Union[str, Path]]] = None,
        min_track_length: int = 10,
        force_feature_recalc: bool = False,
        min_overlapping_ratio: int = 1,
        trajectory_min_iou_thresh: float = 0.4,
        trajectory_min_overlap_ratio: float = 0.3,
        trajectory_max_iou_track_distance: float = 0.6,
    ):
        self.test_all_measurements_gt_pairs_secondary = None
        self.all_measurements_gt_pairs_secondary = None
        self.all_measurements_gt_pairs = None
        self.test_all_measurements_gt_pairs = None
        self.measurements_dfs = None
        self.test_dfs = None
        self.min_overlapping_ratio = min_overlapping_ratio
        self.min_track_length = min_track_length
        self.measurements_csv_paths = [Path(p) for p in measurements_csv_paths]
        self.test_csv_paths = [Path(p) for p in test_csv_paths]
        self.force_feature_recalc = force_feature_recalc
        self.trajectory_min_iou_thresh = trajectory_min_iou_thresh
        self.trajectory_min_overlap_ratio = trajectory_min_overlap_ratio
        self.trajectory_max_iou_track_distance = trajectory_max_iou_track_distance
        gt_csv_paths = [Path(p) for p in gt_csv_paths]
        if test_gt_csv_paths:
            self.test_gt_csv_paths = [Path(p) for p in test_gt_csv_paths]
        else:
            self.test_gt_csv_paths = None
        self.masks = {
            "rake_mask": self._read_mask(rake_mask_path),
            "flow_area_mask": self._read_mask(flow_area_mask_path),
            "non_flow_area_mask": self._read_mask(non_flow_area_mask_path),
        }
        if load_old_measurements:
            self.measurements_dfs, self.gt_dfs, self.test_cached_csv_paths = self._load_old_measurements_df(
                self.measurements_csv_paths, gt_csv_paths
            )
        else:
            self.measurements_dfs, self.gt_dfs, self.cached_csv_paths = self._load_measurements_df(
                self.measurements_csv_paths, gt_csv_paths
            )
        if load_old_test_files:
            self.test_dfs, self.test_gt_dfs, self.test_cached_csv_paths = self._load_old_measurements_df(
                self.test_csv_paths, test_gt_csv_paths
            )
        else:
            self.test_dfs, self.test_gt_dfs, self.test_cached_csv_paths = self._load_measurements_df(
                self.test_csv_paths, test_gt_csv_paths
            )

    def format_old_classifications(self, labels_dfs: list[pd.DataFrame]):
        dfs = labels_dfs.copy()
        for idx, df in enumerate(dfs):
            dfs[idx]["assigned_label"] = df["classification"].apply(lambda x: 1 if x == "fish" else 0)
            feature_df = (
                dfs[idx]
                .groupby("id")
                .apply(lambda x: calculate_flow_area_time_ratio(x, self.masks["flow_area_mask"]))
                .to_frame("flow_area_time_ratio")
            )
            dfs[idx] = dfs[idx].join(feature_df, on="id", how="left")
            dfs[idx]["video_id"] = idx
            dfs[idx]["id"] = dfs[idx].apply(lambda x: f"{x['video_id']}-{x['id']}", axis=1)
        return dfs

    def _load_measurements_df(
        self,
        labels_csv_paths: list[Path],
        gt_csv_paths: list[Path],
    ) -> tuple[list[Union[DataFrame, DataFrame]], list[Union[DataFrame, DataFrame]], list[Path]]:
        labels_dfs, gt_dfs, cache_paths = [], [], []
        for idx, (labels_df, gt_df, cache_path) in enumerate(self._read_csvs(labels_csv_paths, gt_csv_paths)):
            if not gt_df.empty:
                gt_df["id"] = gt_df["id"].apply(lambda x: f"{idx}-{x}")
                gt_dfs.append(gt_df)
            labels_df["video_id"] = idx
            labels_df["id"] = labels_df.apply(lambda x: f"{x['video_id']}-{x['id']}", axis=1)
            labels_dfs.append(labels_df)
            cache_paths.append(cache_path)
        return labels_dfs, gt_dfs, cache_paths

    def _load_old_measurements_df(
        self,
        labels_csv_paths: list[Path],
        gt_csv_paths: list[Path],
    ) -> tuple[list[Union[DataFrame, DataFrame]], list[Union[DataFrame, DataFrame]], list[Path]]:
        labels_dfs = self.read_csvs_from_paths(labels_csv_paths)
        labels_dfs = self.format_old_classifications(labels_dfs)
        gt_dfs = self.read_csvs_from_paths(gt_csv_paths)
        cache_paths = [self._create_cache_path(path) for path in labels_csv_paths]
        return labels_dfs, gt_dfs, cache_paths

    def calc_feature_dfs(self):
        if self.force_feature_recalc:
            if self.test_dfs:
                test_df_list = []
                print("Calculating/reading features for test data")
                for test_df, cache_path in tqdm(zip(self.test_dfs, self.test_cached_csv_paths)):
                    calculated_test_df = self.calculate_features_on_tracks(cache_path, test_df)
                    test_df_list.append(calculated_test_df)
                self.test_dfs = test_df_list

            measurements_df_list = []
            print("Calculating/reading features for training data")
            for measurements_df, cache_path in tqdm(zip(self.measurements_dfs, self.cached_csv_paths)):
                calculated_measurements_df = self.calculate_features_on_tracks(cache_path, measurements_df)
                measurements_df_list.append(calculated_measurements_df)
            self.measurements_dfs = measurements_df_list

        self.measurements_dfs, self.all_measurements_gt_pairs, self.all_measurements_gt_pairs_secondary = (
            self._map_measurements2gt_trajectory(self.measurements_dfs, self.gt_dfs)
        )
        if len(self.test_gt_dfs) != 0:
            self.test_dfs, self.test_all_measurements_gt_pairs, self.test_all_measurements_gt_pairs_secondary = (
                self._map_measurements2gt_trajectory(self.test_dfs, self.test_gt_dfs)
            )
        else:
            for df in self.test_dfs:
                df["gt_label"] = None

    def _read_csvs(
        self,
        labels_csv_paths: list[Path],
        gt_csv_paths: list[Path],
    ) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, Path]]:
        print("Calculating/reading features")
        for path in tqdm(labels_csv_paths):
            cache_path = self._create_cache_path(path)
            if gt_csv_paths:
                gt_df = pd.read_csv(gt_csv_paths[labels_csv_paths.index(path)], delimiter=",")
            else:
                gt_df = pd.DataFrame()
            if cache_path.exists() and not self.force_feature_recalc:
                print(f"Reading cached features from {cache_path}")
                labels_df = load_csv_with_tiles(cache_path)
            else:
                labels_df = load_csv_with_tiles(path)
                labels_df = self.filter_tracks(labels_df)
            yield labels_df, gt_df, cache_path

    def _create_cache_path(self, path: Path) -> Path:
        return path.with_stem(path.stem + f"_cached_features_min_track_length_{self.min_track_length}").with_suffix(
            ".csv"
        )

    def _create_classification_save_path(self, path: Path) -> Path:
        return path.with_stem(path.stem + f"_classification_min_track_length_{self.min_track_length}").with_suffix(
            ".csv"
        )

    def read_csvs_from_paths(self, csv_paths: list[Path]) -> list[pd.DataFrame]:
        return [pd.read_csv(path, delimiter=",") for path in tqdm(csv_paths)]

    def calculate_features_on_tracks(self, cache_path, labels_df):
        labels_df = calculate_features(labels_df, self.masks)
        labels_df = self.filter_features(labels_df)
        if not labels_df.empty:
            save_df = labels_df.copy()
            save_df["image_tile"] = save_df["image_tile"].apply(lambda x: x.tolist())
            try:
                save_df["raw_image_tile"] = save_df["raw_image_tile"].apply(lambda x: x.tolist())
            except KeyError:
                pass
            save_df.to_csv(cache_path, index=False)
            return labels_df
        else:
            return pd.DataFrame()

    @staticmethod
    def _read_mask(mask_path: Union[str, Path]) -> np.ndarray:
        mask = cv.imread(Path(mask_path).as_posix(), cv.IMREAD_GRAYSCALE)
        assert mask is not None, f"Could not read mask from {mask_path}"
        mask = mask[49:1001, 92:1831]  # Drop the border
        return cv.resize(mask, (480, 270)) > 0

    def _map_measurements2gt_trajectory(
        self, labels_dfs_in: list[pd.DataFrame], gt_dfs_in: list[pd.DataFrame]
    ) -> tuple[DataFrame, dict[Any, Any], dict[Any, Any]]:
        print("Mapping labels to ground truth trajectories")

        labels_dfs = labels_dfs_in.copy()
        gt_dfs = gt_dfs_in.copy()
        all_labels_gt_pairs, all_labels_gt_pairs_secondary = [], []
        for df_idx, (labels_df, gt_df) in enumerate(zip(labels_dfs, gt_dfs)):
            model_detections = labels_df
            ground_truth = gt_df

            track_distances = np.empty((len(model_detections["id"].unique()), len(ground_truth["id"].unique())))

            for labels_idx, track_id in tqdm(enumerate(model_detections["id"].unique())):
                labels_track = model_detections.loc[model_detections["id"] == track_id, ["frame", "x", "y", "w", "h"]]

                for gt_idx, gt_id in enumerate(ground_truth["id"].unique()):
                    gt_track = ground_truth.loc[ground_truth["id"] == gt_id, ["frame", "x", "y", "w", "h"]]
                    track_distances[labels_idx, gt_idx] = calculate_track_distance(
                        labels_track, gt_track, self.trajectory_min_iou_thresh, self.trajectory_min_overlap_ratio
                    )

            labels_gt_pairs, dist_matrix_indices = self._calculate_trajectory_pairs(
                track_distances, model_detections, ground_truth, self.trajectory_max_iou_track_distance
            )

            track_distances_copy = track_distances.copy()
            track_distances_copy[dist_matrix_indices] = 1.0
            labels_gt_pairs_secondary, _ = self._calculate_trajectory_pairs(
                track_distances_copy, model_detections, ground_truth, self.trajectory_max_iou_track_distance
            )

            labels_dfs[df_idx]["gt_label"] = "noise"
            labels_dfs[df_idx].loc[
                labels_df["id"].isin(np.hstack((labels_gt_pairs[:, 0], labels_gt_pairs_secondary[:, 0]))),
                "gt_label",
            ] = "fish"

            all_labels_gt_pairs.extend(labels_gt_pairs)
            all_labels_gt_pairs_secondary.extend(labels_gt_pairs_secondary)

        all_labels_gt_pairs = dict(all_labels_gt_pairs)
        all_labels_gt_pairs_secondary = dict(all_labels_gt_pairs_secondary)

        return labels_dfs, all_labels_gt_pairs, all_labels_gt_pairs_secondary

    def filter_features(self, measurements_df):
        measurements_df = measurements_df[
            (measurements_df["average_overlap_ratio"] > self.min_overlapping_ratio)
            # & (measurements_df["average_pixel_intensity"] > 127.80)
        ]
        return measurements_df

    def filter_tracks(self, measurements_df):
        value_counts_model = (
            pd.DataFrame(measurements_df.id.value_counts())
            .reset_index()
            .rename(columns={"index": "id", "id": "occurences"})
        )
        measurements_df = measurements_df[
            measurements_df["id"].isin(value_counts_model[value_counts_model.occurences >= self.min_track_length]["id"])
        ]
        return measurements_df

    @staticmethod
    def _calculate_trajectory_pairs(
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
        plot_test_data: bool = False,
        show_track_id: bool = False,
    ) -> None:
        if plot_test_data:
            if "assigned_label" not in self.test_dfs[0].columns:
                raise ValueError(
                    "No clustering has been performed yet. Please perform clustering with the do_clustering method."
                )
            if len(self.test_gt_dfs) != 0:
                ground_truth = pd.concat(self.test_gt_dfs)
            else:
                ground_truth = pd.DataFrame()
            stacked_labels_dfs = self.stacked_test_dfs
            all_measurements_gt_pairs = self.test_all_measurements_gt_pairs
            all_measurements_gt_pairs_secondary = self.test_all_measurements_gt_pairs_secondary
        else:
            if "assigned_label" not in self.measurements_dfs[0].columns:
                raise ValueError(
                    "No clustering has been performed yet. Please perform clustering with the do_clustering method."
                )
            ground_truth = pd.concat(self.gt_dfs)
            stacked_labels_dfs = self.stacked_dfs
            all_measurements_gt_pairs = self.all_measurements_gt_pairs
            all_measurements_gt_pairs_secondary = self.all_measurements_gt_pairs_secondary

        n_labels = stacked_labels_dfs["assigned_label"].nunique()
        show_legend = True if n_labels <= 6 else False
        colormap = cm.get_cmap("viridis", n_labels + 1)  # +1 for the ground truth color

        _, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(10, 10))
        if mask_to_show:
            ax.imshow(self.masks[mask_to_show], cmap="gray", alpha=0.2)
        plt.gca().invert_yaxis()

        if ground_truth.empty:
            for measurements_track_id in stacked_labels_dfs.id.unique():
                self.plot_tracks_and_annotations(
                    ax, colormap, measurements_track_id, stacked_labels_dfs, metric_to_show, show_track_id
                )
            self.create_plot_components(
                ax,
                colormap,
                n_labels,
                show_legend,
                title="Trajectories with assigned label",
                type_of_label="Manual",
            )
        else:
            self.plot_assigned_tracks(
                all_measurements_gt_pairs,
                all_measurements_gt_pairs_secondary,
                ax,
                colormap,
                ground_truth,
                metric_to_show,
                n_labels,
                show_legend,
                show_track_id,
                stacked_labels_dfs,
            )

            self.plot_unassigned_tracks(
                all_measurements_gt_pairs,
                all_measurements_gt_pairs_secondary,
                colormap,
                mask_to_show,
                metric_to_show,
                n_labels,
                show_legend,
                show_track_id,
                stacked_labels_dfs,
            )

    def plot_unassigned_tracks(
        self,
        all_measurements_gt_pairs,
        all_measurements_gt_pairs_secondary,
        colormap,
        mask_to_show,
        metric_to_show,
        n_labels,
        show_legend,
        show_track_id,
        stacked_labels_dfs,
    ):
        _, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(10, 10))
        if mask_to_show:
            ax.imshow(self.masks[mask_to_show], cmap="gray", alpha=0.2)
        plt.gca().invert_yaxis()
        plt.gca().invert_yaxis()
        for measurements_track_id in stacked_labels_dfs.id.unique():
            if (measurements_track_id not in all_measurements_gt_pairs.keys()) and (
                measurements_track_id not in all_measurements_gt_pairs_secondary.keys()
            ):
                self.plot_tracks_and_annotations(
                    ax, colormap, measurements_track_id, stacked_labels_dfs, metric_to_show, show_track_id
                )
        self.create_plot_components(
            ax,
            colormap,
            n_labels,
            show_legend,
            title="non-matched trajectories with assigned label",
            type_of_label="",
        )

    def plot_assigned_tracks(
        self,
        all_measurements_gt_pairs,
        all_measurements_gt_pairs_secondary,
        ax,
        colormap,
        ground_truth,
        metric_to_show,
        n_labels,
        show_legend,
        show_track_id,
        stacked_labels_dfs,
    ):
        for measurements_track_id, gt_track_id in all_measurements_gt_pairs.items():
            self.plot_tracks_and_annotations(
                ax, colormap, measurements_track_id, stacked_labels_dfs, metric_to_show, show_track_id
            )
            if not ground_truth.empty:
                gt_track_df = ground_truth[ground_truth.id == gt_track_id]
                ax.plot(gt_track_df.x, gt_track_df.y, alpha=0.5, color=colormap(0), linestyle="dashed")
        for measurements_track_id, gt_track_id in all_measurements_gt_pairs_secondary.items():
            self.plot_tracks_and_annotations(
                ax, colormap, measurements_track_id, stacked_labels_dfs, metric_to_show, show_track_id
            )
        self.create_plot_components(
            ax,
            colormap,
            n_labels,
            show_legend,
            title="matched trajectories with assigned label",
            type_of_label="Assigned",
        )

    def create_plot_components(self, ax, colormap, n_labels, show_legend, title, type_of_label):
        ax.set(ylabel="y", title=title, ylim=[270, 0], xlim=[0, 480])
        ax.set_aspect("equal", adjustable="box")
        legend_elements = [Line2D([0], [0], color=colormap(0), lw=2, label=f"{type_of_label} label")]
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
        show_track_id: bool = False,
    ) -> None:
        track_df = model_detections[model_detections.id == measurements_track_id]
        ax.plot(
            track_df.x,
            track_df.y,
            color=colormap(int(track_df.assigned_label.iloc[0]) + 1),
        )
        metric_annotation = str(track_df[metric_to_show].iloc[0])[:6] if metric_to_show else ""
        if show_track_id:
            ax.annotate(
                f"{measurements_track_id}, {metric_annotation}",
                (track_df.x.iloc[0], track_df.y.iloc[0]),
                fontsize=5,
            )

    def plot_image_tiles_along_trajectory(
        self,
        measurements_track_id: int,
        every_nth_frame: int,
        feature_names: list[str] = ["image_tile"],
    ) -> None:
        track_df = self._get_track_df_by_id(measurements_track_id)

        for idx, row in track_df.iterrows():
            if idx % every_nth_frame == 0:
                for feature_name in feature_names:
                    image_tile = np.squeeze(row[feature_name])
                    if image_tile.shape[-1] == 3:
                        plt.imshow(image_tile[:, :, ::-1])
                    else:
                        plt.imshow(image_tile, cmap="gray")
                    plt.title(f"Frame {row.frame} - {feature_name}")
                    plt.show()

    def show_trajectory_numeric_features(
        self,
        measurements_track_ids: list[int],
        boxplot_split_thresholds: list[float] = [1],
    ) -> None:
        boxplot_split_thresholds = sorted(boxplot_split_thresholds)
        features_to_print = [feature for feature in self.feature_names if feature not in ["image_tile", "video_id"]]
        features_to_plot = [
            feature for feature in features_to_print if feature not in ["classification", "gt_label", "assigned_label"]
        ]

        all_tracks_df = self.stacked_dfs.groupby(["video_id", "id"]).first().reset_index()

        # Calculate medians
        medians = all_tracks_df[features_to_plot].median()

        # Split features based on thresholds
        subplots = len(boxplot_split_thresholds) + 1
        _, axs = plt.subplots(subplots, figsize=(10, subplots * 7))  # Increase the figure size

        for measurements_track_id in measurements_track_ids:
            track_df = self._get_track_df_by_id(measurements_track_id).iloc[0]

            for i, threshold in enumerate(boxplot_split_thresholds):
                if i == 0:
                    features_above_threshold = [
                        feature for feature in features_to_plot if medians[feature] <= threshold
                    ]
                    axs[i].set_title(f"Numeric Features Distribution (Median <= {threshold})")
                    axs[i].boxplot(all_tracks_df[features_above_threshold].values, labels=features_above_threshold)
                    axs[i].plot(
                        range(1, len(features_above_threshold) + 1),
                        track_df[features_above_threshold].values.tolist(),
                        "ro",
                    )
                else:
                    features_above_threshold = []
                    for feature in features_to_plot:
                        if (
                            medians[feature] > boxplot_split_thresholds[i - 1]
                            and medians[feature] <= boxplot_split_thresholds[i]
                        ):
                            features_above_threshold.append(feature)
                    axs[i].set_title(
                        f"Features Distr ({boxplot_split_thresholds[i - 1]} < Median <= {boxplot_split_thresholds[i]})"
                    )
                    axs[i].boxplot(all_tracks_df[features_above_threshold].values, labels=features_above_threshold)
                    axs[i].plot(
                        range(1, len(features_above_threshold) + 1),
                        track_df[features_above_threshold].values.tolist(),
                        "ro",
                    )
                    if i == len(boxplot_split_thresholds) - 1:
                        features_above_threshold = [
                            feature for feature in features_to_plot if medians[feature] > threshold
                        ]
                        axs[i + 1].set_title(f"Numeric Features Distribution (Median > {threshold})")
                        axs[i + 1].boxplot(
                            all_tracks_df[features_above_threshold].values, labels=features_above_threshold
                        )
                        axs[i + 1].plot(
                            range(1, len(features_above_threshold) + 1),
                            track_df[features_above_threshold].values.tolist(),
                            "ro",
                        )

        for ax in axs:
            ax.set_xlabel("Features")
            ax.set_ylabel("Values")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        plt.subplots_adjust(hspace=0.8)  # Adjust the value as needed
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

        for idx, df in enumerate(self.test_dfs):
            X_test = scaler.transform(df[features])
            self.test_dfs[idx]["assigned_label"] = clustering.predict(X_test)

    def do_binary_classification(
        self,
        model: Callable,
        features: list[str],
        features_flow_area: list[str],
        kfold_n_splits: int,
        models: dict[str, Any] = [],
        scalers: dict[str, Any] = [],
        manual_noise_thresholds: Optional[tuple[str, str, float]] = None,
        manual_noise_thresholds_flow_area: Optional[tuple[str, str, float]] = None,
    ):
        if features_flow_area is not None:
            if len(models) == 0 and len(scalers) == 0:
                df, models, scalers = self.perform_flow_area_classification(
                    features,
                    features_flow_area,
                    kfold_n_splits,
                    manual_noise_thresholds,
                    manual_noise_thresholds_flow_area,
                    model,
                )
                self.save_model_and_scaler_pickles(
                    models=models,
                    scalers=scalers,
                )
                self.dump_manual_noise_thresholds_with_models_and_scalers_to_json(
                    manual_noise_thresholds,
                    manual_noise_thresholds_flow_area,
                )
                self.merge_assigned_labels_to_measurements(df)
            elif len(models) == 1 and len(scalers) == 1:
                if "non_flow" in models and "non_flow" in scalers:
                    models["flow"] = NoFeatureModel()
                    scalers["flow"] = NoFeatureScaler()
                else:
                    raise ValueError("Invalid model and scaler input")
            df_test = self.score_test_data(
                features,
                features_flow_area,
                manual_noise_thresholds,
                manual_noise_thresholds_flow_area,
                models,
                scalers,
            )
        else:
            # TODO rain model for all trajectories if features_flow_area is None
            raise NotImplementedError("Not supported anymore")

        if self.test_dfs:
            for idx, test_df in enumerate(self.test_dfs):
                try:
                    test_df.drop(columns=["assigned_label"], inplace=True)
                except KeyError:
                    pass
                video_id = test_df["video_id"].iloc[0]
                right_df = df_test[df_test["video_id"] == video_id][["id", "assigned_label"]]
                self.test_dfs[idx] = test_df.merge(right_df, on="id", how="left")

    def merge_assigned_labels_to_measurements(self, df):
        for idx, measurement_df in enumerate(self.measurements_dfs):
            try:
                measurement_df.drop(columns=["assigned_label"], inplace=True)
            except KeyError:
                pass
            video_id = measurement_df["video_id"].iloc[0]
            right_df = df[df["video_id"] == video_id][["id", "assigned_label"]]
            self.measurements_dfs[idx] = measurement_df.merge(right_df, on="id", how="left")

    def perform_flow_area_classification(
        self,
        features,
        features_flow_area,
        kfold_n_splits,
        manual_noise_thresholds,
        manual_noise_thresholds_flow_area,
        model,
    ):
        df = self.stacked_dfs.groupby("id").first().reset_index()
        flow_area_indices = df["flow_area_time_ratio"] > 0.5
        df.loc[flow_area_indices, "assigned_label"], model_flow, scaler_flow = (
            self._do_binary_classification_for_trajectory_subset(
                df[flow_area_indices], model, features_flow_area, kfold_n_splits
            )
        )
        if manual_noise_thresholds_flow_area:
            df.loc[flow_area_indices, "assigned_label"], _ = self._filter_with_manual_thresholds(
                df[flow_area_indices], manual_noise_thresholds_flow_area
            )
        df.loc[~flow_area_indices, "assigned_label"], model_non_flow, scaler = (
            self._do_binary_classification_for_trajectory_subset(
                df[~flow_area_indices], model, features, kfold_n_splits
            )
        )
        if manual_noise_thresholds:
            df.loc[~flow_area_indices, "assigned_label"], _ = self._filter_with_manual_thresholds(
                df[~flow_area_indices], manual_noise_thresholds
            )
        models = {"flow": model_flow, "non_flow": model_non_flow}
        scalers = {"flow": scaler_flow, "non_flow": scaler}
        return df, models, scalers

    def save_model_and_scaler_pickles(self, models: dict, scalers: dict):
        saved_models = []
        saved_scalers = []
        for key, model in models.items():
            if not isinstance(model, NoFeatureModel):
                model_pkl_file = f"{key}_classifier_model.pkl"
                with open(model_pkl_file, "wb") as file:
                    saved_models.append(model_pkl_file)
                    pickle.dump(model, file)
        for key, scaler in scalers.items():
            if not isinstance(scaler, NoFeatureScaler):
                scaler_pkl_file = f"{key}_scaler.pkl"
                with open(scaler_pkl_file, "wb") as file:
                    saved_scalers.append(scaler_pkl_file)
                    pickle.dump(scaler, file)
        self.saved_models = saved_models
        self.saved_scalers = saved_scalers

    def score_test_data(
        self,
        features,
        features_flow_area,
        manual_noise_thresholds,
        manual_noise_thresholds_flow_area,
        models,
        scalers,
    ):
        if self.test_dfs:
            df_test = self.stacked_test_dfs.groupby("id").first().reset_index()
            flow_area_indices = df_test["flow_area_time_ratio"] > 0.5
            df_test.loc[flow_area_indices, "assigned_label"] = models["flow"].predict(
                scalers["flow"].transform(df_test[flow_area_indices][features_flow_area])
            )
            if manual_noise_thresholds_flow_area:
                df_test.loc[flow_area_indices, "assigned_label"], _ = self._filter_with_manual_thresholds(
                    df_test[flow_area_indices], manual_noise_thresholds_flow_area
                )
            df_test.loc[~flow_area_indices, "assigned_label"] = models["non_flow"].predict(
                scalers["non_flow"].transform(df_test[~flow_area_indices][features])
            )
            if manual_noise_thresholds:
                df_test.loc[~flow_area_indices, "assigned_label"], _ = self._filter_with_manual_thresholds(
                    df_test[~flow_area_indices], manual_noise_thresholds
                )
        return df_test

    @staticmethod
    def _filter_with_manual_thresholds(
        df_in: pd.DataFrame, manual_thresholds: tuple[str, str, float]
    ) -> tuple[pd.Series, np.array]:
        df = df_in.copy()
        removed_indices = np.zeros(len(df))
        for feature, operator, threshold in manual_thresholds:
            if operator == "smaller":
                df.loc[df[feature] < threshold, "assigned_label"] = 0
                removed_indices = np.logical_or(removed_indices, df[feature] < threshold)
            elif operator == "larger":
                df.loc[df[feature] > threshold, "assigned_label"] = 0
                removed_indices = np.logical_or(removed_indices, df[feature] > threshold)
            else:
                raise ValueError(f"Invalid operator: {operator}")
        return df["assigned_label"], removed_indices

    @staticmethod
    def _do_binary_classification_for_trajectory_subset(
        df: pd.DataFrame,
        model: Callable,
        features: list[str],
        kfold_n_splits: int,
    ) -> tuple[np.array, Callable, preprocessing.StandardScaler]:
        if not features:
            y_kfold = np.ones(df.shape[0])
            model_all = NoFeatureModel()
            scaler_all = NoFeatureScaler()
        else:
            X = np.array(df[features])
            y = np.array([1 if lbl == "fish" else 0 for lbl in df["gt_label"]])

            kf = KFold(n_splits=kfold_n_splits)
            y_kfold = np.empty(y.shape)
            for train_index, val_index in kf.split(X):
                X_train, X_val = X[train_index], X[val_index]
                scaler = preprocessing.StandardScaler().fit(X_train)
                X_train, X_val = scaler.transform(X_train), scaler.transform(X_val)
                y_train, _ = y[train_index], y[val_index]

                model = deepcopy(model).fit(X_train, y_train)
                y_kfold[val_index] = model.predict(X_val)

            scaler_all = preprocessing.StandardScaler().fit(X)
            X = scaler_all.transform(X)
            model_all = deepcopy(model).fit(X, y)

        return y_kfold, model_all, scaler_all

    @staticmethod
    def worker(args):
        self, model, features, kfold_n_splits, distinguish_flow_areas = args
        self.do_binary_classification(model, features, kfold_n_splits, distinguish_flow_areas)
        _, _, _, f1, _ = self.calculate_metrics()
        return features, f1

    def sweep_classification_feature_selection(
        self,
        model: Callable,
        kfold_n_splits: int = 5,
        max_n_features: int = 3,
        distinguish_flow_areas: bool = False,
    ):
        all_features = [
            feat
            for feat in self.feature_names
            if feat
            not in [
                "image_tile",
                "raw_image_tile",
                "video_id",
                "classification",
                "gt_label",
                "assigned_label",
                "binary_image",
            ]
        ]

        args_list = []
        for n_features in range(1, max_n_features + 1):
            for features in itertools.combinations(all_features, n_features):
                args_list.append((self, model, list(features), kfold_n_splits, distinguish_flow_areas))

        with Pool(cpu_count()) as p:
            results = p.map(self.worker, args_list)

        best_features, best_score = max(results, key=lambda x: x[1])
        return best_features, best_score

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

    def calculate_metrics(
        self,
        beta_vals: list[int] = [2],
        make_plots: bool = False,
        verbosity: int = 0,
        evaluate_test_data: bool = False,
        distinguish_flow_areas: bool = False,
        normalize_confusion_matrix: str = None,
    ) -> tuple[np.array, float, float, float, list[float]]:
        # TODO: group by video_id and id
        if evaluate_test_data:
            df = self.stacked_test_dfs.groupby(["id"]).first().reset_index()
        else:
            df = self.stacked_dfs.groupby(["id"]).first().reset_index()

        df["gt_label"] = df["gt_label"].apply(lambda x: 1 if x == "fish" else 0)

        if distinguish_flow_areas:
            flow_area_indices = df["flow_area_time_ratio"] > 0.5
            input_dict = {
                "flow_area": [
                    df.loc[flow_area_indices, "gt_label"],
                    df.loc[flow_area_indices, "assigned_label"],
                ],
                "non_flow_area": [
                    df.loc[~flow_area_indices, "gt_label"],
                    df.loc[~flow_area_indices, "assigned_label"],
                ],
            }
        else:
            input_dict = {
                "all": [
                    df["gt_label"],
                    df["assigned_label"],
                ]
            }

        metrics_dict = {}
        for area, inputs in input_dict.items():
            confusion_matrix = metrics.confusion_matrix(*inputs, normalize=normalize_confusion_matrix, labels=[0, 1])
            precision = metrics.precision_score(*inputs, labels=[0, 1], average="binary")
            recall = metrics.recall_score(*inputs, labels=[0, 1], average="binary")
            f1 = metrics.f1_score(*inputs, labels=[0, 1], average="binary")
            if verbosity >= 1:
                print(f"Metrics for {area}")
                print(f"Precision: {precision}")
                print(f"Recall: {recall}")
                print(f"F1 score: {f1}")
            fbeta = []
            for beta in beta_vals:
                fbeta.append(metrics.fbeta_score(*inputs, beta=beta, labels=[0, 1], average="binary"))
                if verbosity >= 1:
                    print(f"F{beta} score: {fbeta[-1]}")
            if verbosity >= 1:
                if make_plots:
                    cm_display = metrics.ConfusionMatrixDisplay(
                        confusion_matrix, display_labels=["noise", "fish"]
                    ).plot()
                    cm_display.ax_.set_title(f"Confusion matrix for {area}")
            else:
                print(f"Confusion matrix: {confusion_matrix}")
            metrics_dict[area] = {
                "confusion_matrix": confusion_matrix,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "fbeta": fbeta,
            }

        return metrics_dict

    def save_classified_tracks_to_csv(self):
        for path, df in zip(self.test_csv_paths + self.measurements_csv_paths, self.test_dfs + self.measurements_dfs):
            save_path = self._create_classification_save_path(path)
            save_df = df.copy()
            save_df.drop(columns=["image_tile", "raw_image_tile"], inplace=True, errors="ignore")
            save_df.to_csv(save_path, index=False)

    def dump_manual_noise_thresholds_with_models_and_scalers_to_json(
        self, manual_noise_thresholds, manual_noise_thresholds_flow_area
    ):
        json_to_save = {
            "manual_noise_thresholds": manual_noise_thresholds,
            "manual_noise_thresholds_flow_area": manual_noise_thresholds_flow_area,
            "saved_models": self.saved_models,
            "saved_scalers": self.saved_scalers,
        }
        with open("classification_parameters.json", "w") as file:
            json.dump(json_to_save, file)

    @property
    def feature_names(self):
        return self.measurements_dfs[0].columns.tolist()[9:]

    @property
    def stacked_dfs(self):
        return pd.concat(self.measurements_dfs)

    @property
    def stacked_test_dfs(self):
        return pd.concat(self.test_dfs)


class NoFeatureModel(object):

    def __init__(self):
        pass

    def predict(self, X):
        return np.ones(X.shape[0])


class NoFeatureScaler(object):

    def __init__(self):
        pass

    def transform(self, X):
        return X
