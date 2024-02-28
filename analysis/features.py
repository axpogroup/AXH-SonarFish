import json
from pathlib import Path
from typing import Callable, Iterator, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from motmetrics.distances import boxiou
from scipy.optimize import linear_sum_assignment
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


class FeatureGenerator(object):
    def __init__(
        self,
        measurements_csv_paths: list[Union[str, Path]],
        gt_csv_paths: list[Union[str, Path]],
        min_track_length: int = 10,
        force_feature_recalc: bool = False,
    ):
        self.min_track_length = min_track_length
        measurements_csv_paths = [Path(p) for p in measurements_csv_paths]
        gt_csv_paths = [Path(p) for p in gt_csv_paths]
        self._calc_feature_dfs(measurements_csv_paths, gt_csv_paths, force_feature_recalc)

    def _calc_feature_dfs(
        self, measurements_csv_paths: list[Path], gt_csv_paths: list[Path], force_feature_recalc: bool = False
    ) -> None:
        self.measurements_dfs, self.gt_dfs = [], []
        for measurements_df, gt_df, is_cached, cache_path in self._read_csvs(
            measurements_csv_paths, gt_csv_paths, force_feature_recalc
        ):
            self.gt_dfs.append(gt_df)
            if is_cached and not force_feature_recalc:
                self.measurements_dfs.append(measurements_df)
            else:
                measurements_df = calculate_features(measurements_df)
                measurements_df.to_csv(cache_path, index=False)
                self.measurements_dfs.append(measurements_df)

    def _read_csvs(
        self,
        measurements_csv_paths: list[Path],
        gt_csv_paths: list[Path],
        force_feature_recalc: bool = False,
    ) -> Iterator[dict[str, Union[pd.DataFrame, bool, Path]]]:
        for path in measurements_csv_paths:
            cache_path = path.with_stem(
                path.stem + f"_cached_features_min_track_length_{self.min_track_length}"
            ).with_suffix(".csv")
            gt_df = pd.read_csv(gt_csv_paths[measurements_csv_paths.index(path)], delimiter=",")
            if cache_path.exists() and not force_feature_recalc:
                print(f"Reading cached features from {cache_path}")
                measurements_df = load_csv_with_tiles(cache_path)
                yield measurements_df, gt_df, True, cache_path
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
                measurements_df.to_csv(cache_path, index=False)
                yield measurements_df, gt_df, False, cache_path

    def map_measurements2gt_trajectory(
        self, min_iou_thresh: float = 0.4, min_overlap_ratio: float = 0.3, max_iou_track_distance: float = 0.6
    ) -> tuple[dict[int, int], dict[int, int]]:
        all_measurements_gt_pairs = []
        all_measurements_gt_pairs_secondary = []

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

        return dict(all_measurements_gt_pairs), dict(all_measurements_gt_pairs_secondary)

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
        min_iou_thresh: float = 0.4,
        min_overlap_ratio: float = 0.3,
        max_iou_track_distance: float = 0.6,
    ) -> None:
        if "cluster" not in self.measurements_dfs[0].columns:
            raise ValueError(
                "No clustering has been performed yet. Please perform clustering with the do_clustering method."
            )

        all_measurements_gt_pairs, all_measurements_gt_pairs_secondary = self.map_measurements2gt_trajectory(
            min_iou_thresh, min_overlap_ratio, max_iou_track_distance
        )

        model_detections = pd.concat(self.measurements_dfs)
        ground_truth = pd.concat(self.gt_dfs)

        n_clusters = model_detections["cluster"].nunique()
        colormap = cm.get_cmap("viridis", n_clusters + 1)  # +1 for the ground truth color

        _, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(10, 10))
        plt.gca().invert_yaxis()

        # Plot assigned tracks
        for measurements_track_id, gt_track_id in all_measurements_gt_pairs.items():
            self.plot_tracks_and_annotations(ax, colormap, measurements_track_id, model_detections)
            gt_track_df = ground_truth[ground_truth.id == gt_track_id]
            ax.plot(gt_track_df.x, gt_track_df.y, alpha=0.5, color=colormap(0), linestyle="dashed")

        for measurements_track_id, gt_track_id in all_measurements_gt_pairs_secondary.items():
            measurements_track_df = model_detections[model_detections.id == measurements_track_id]
            ax.plot(
                measurements_track_df.x,
                measurements_track_df.y,
                color=colormap(measurements_track_df.cluster.iloc[0] + 1),
                linestyle="dotted",
            )

        ax.set(ylabel="y", title="assigned trajectories with clustering", ylim=[270, 0], xlim=[0, 480])
        ax.set_aspect("equal", adjustable="box")

        legend_elements = [Line2D([0], [0], color=colormap(0), lw=2, label="Label")]
        for i in range(n_clusters):
            legend_elements.append(Line2D([0], [0], color=colormap(i + 1), lw=2, label=f"Cluster {i}"))
        ax.legend(handles=legend_elements)

        plt.show()

        # Plot tracks that were not assigned to a ground truth
        _, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(10, 10))
        plt.gca().invert_yaxis()

        for measurements_track_id in model_detections.id.unique():
            if (measurements_track_id not in all_measurements_gt_pairs.keys()) and (
                measurements_track_id not in all_measurements_gt_pairs_secondary.keys()
            ):
                self.plot_tracks_and_annotations(ax, colormap, measurements_track_id, model_detections)
        ax.set(ylabel="y", title="unassigned trajectories with clustering", ylim=[270, 0], xlim=[0, 480])
        ax.set_aspect("equal", adjustable="box")

        legend_elements = [Line2D([0], [0], color=colormap(0), lw=2, label="Label")]
        for i in range(n_clusters):
            legend_elements.append(Line2D([0], [0], color=colormap(i + 1), lw=2, label=f"Cluster {i}"))
        ax.legend(handles=legend_elements)

        plt.show()

        return all_measurements_gt_pairs, all_measurements_gt_pairs_secondary

    def plot_tracks_and_annotations(self, ax, colormap, measurements_track_id, model_detections):
        measurements_track_df = model_detections[model_detections.id == measurements_track_id]
        ax.plot(
            measurements_track_df.x,
            measurements_track_df.y,
            color=colormap(measurements_track_df.cluster.iloc[0] + 1),
        )
        ax.annotate(
            f"{measurements_track_id}, {str(measurements_track_df.average_overlap_ratio.iloc[0])[:4]}",
            (measurements_track_df.x.iloc[0], measurements_track_df.y.iloc[0]),
        )

    def do_clustering(self, features: list[str], clustering_method: Callable, n_clusters: int):
        labels = []
        for idx in range(len(self.measurements_dfs)):
            selected_features = pd.concat(self.measurements_dfs)[features]
            clustering = clustering_method(n_clusters=n_clusters)
            labels.append(clustering.fit_predict(selected_features))
            self.measurements_dfs[idx]["cluster"] = labels[-1]
        return labels


def load_csv_with_tiles(path: Path) -> pd.DataFrame:
    csv_with_tiles_df = pd.read_csv(path, delimiter=",")
    csv_with_tiles_df["image_tile"] = csv_with_tiles_df["image_tile"].apply(lambda x: np.array(json.loads(x)))
    return csv_with_tiles_df
