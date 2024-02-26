from pathlib import Path
from typing import Callable, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from motmetrics.distances import boxiou
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


def trace_window_metrics(group):
    # Calculate the Euclidean distance between previous and current positions
    x_diff = np.diff(group["x"])
    y_diff = np.diff(group["y"])
    euclidean_distances = np.sqrt(x_diff**2 + y_diff**2)

    # Calculate the sum of distances over the group
    traversed_distance = np.sum(euclidean_distances)

    # Calculate the frame number difference
    frame_diff = group["frame"].iloc[-1] - group["frame"].iloc[0]

    return pd.Series({"traversed_distance": traversed_distance, "frame_diff": frame_diff})


class FeatureGenerator(object):
    def __init__(
        self,
        mmt_csv_paths: list[Union[str, Path]],
        gt_csv_paths: list[Union[str, Path]],
        min_track_length: int = 10,
    ):
        self.min_track_length = min_track_length
        mmt_csv_paths = [Path(p) for p in mmt_csv_paths]
        gt_csv_paths = [Path(p) for p in gt_csv_paths]
        self._calc_feature_dfs(mmt_csv_paths, gt_csv_paths)

    def _calc_feature_dfs(self, mmt_csv_paths: list[Path], gt_csv_paths: list[Path]):
        self.mmt_dfs, self.gt_dfs = [], []
        for mmt_df, gt_df, is_cached, cache_pth in self._read_csvs(mmt_csv_paths, gt_csv_paths):
            self.gt_dfs.append(gt_df)
            if is_cached:
                self.mmt_dfs.append(mmt_df)
            else:
                mmt_df = self._calculate_features(mmt_df)
                mmt_df.to_csv(cache_pth, index=False)
                self.mmt_dfs.append(mmt_df)

    def _read_csvs(self, mmt_csv_paths: list[Path], gt_csv_paths: list[Path]):
        for pth in mmt_csv_paths:
            cache_pth = pth.with_stem(
                pth.stem + f"_cached_features_min_track_length_{self.min_track_length}"
            ).with_suffix(".csv")
            if cache_pth.exists():
                print(f"Reading cached features from {cache_pth}")
                mmt_df = pd.read_csv(cache_pth, delimiter=",")
                gt_df = pd.read_csv(gt_csv_paths[mmt_csv_paths.index(pth)], delimiter=",")
                yield mmt_df, gt_df, True, cache_pth
            else:
                mmt_df = pd.read_csv(pth, delimiter=",")
                value_counts_model = (
                    pd.DataFrame(mmt_df.id.value_counts())
                    .reset_index()
                    .rename(columns={"index": "id", "id": "occurences"})
                )
                mmt_df = mmt_df[
                    mmt_df["id"].isin(value_counts_model[value_counts_model.occurences >= self.min_track_length]["id"])
                ]
                mmt_df.to_csv(cache_pth, index=False)
                gt_df = pd.read_csv(gt_csv_paths[mmt_csv_paths.index(pth)], delimiter=",")
                yield mmt_df, gt_df, False, cache_pth

    @staticmethod
    def _calculate_features(mmt_df: pd.DataFrame):
        feature_df = mmt_df.groupby("id").apply(trace_window_metrics)
        return mmt_df.join(feature_df, on="id", how="left")

    def map_mmt2gt(
        self, min_iou_thresh: float = 0.4, min_overlap_ratio: float = 0.3, max_iou_track_distance: float = 0.6
    ) -> tuple[dict[int, int], dict[int, int]]:
        all_mmt_gt_pairs, all_mmt_gt_pairs_secondary = [], []

        for mmt_df, gt_df in zip(self.mmt_dfs, self.gt_dfs):
            model_detections = mmt_df
            ground_truth = gt_df

            track_distances = np.empty((len(model_detections["id"].unique()), len(ground_truth["id"].unique())))

            for mmt_idx, track_id in tqdm(enumerate(model_detections["id"].unique())):
                mmt_track = model_detections.loc[model_detections["id"] == track_id, ["frame", "x", "y", "w", "h"]]

                for gt_idx, gt_id in enumerate(ground_truth["id"].unique()):
                    gt_track = ground_truth.loc[ground_truth["id"] == gt_id, ["frame", "x", "y", "w", "h"]]

                    if set(mmt_track["frame"]).isdisjoint(set(gt_track["frame"])):
                        mmt_to_gt_dist = 1.0
                    else:
                        joined = mmt_track.set_index("frame").join(
                            gt_track.set_index("frame"), how="outer", lsuffix="_mmt", rsuffix="_gt"
                        )
                        iou = boxiou(
                            joined[["x_mmt", "y_mmt", "w_mmt", "h_mmt"]].values,
                            joined[["x_gt", "y_gt", "w_gt", "h_gt"]].values,
                        )
                        iou[iou < min_iou_thresh] = np.nan
                        iou_cost = 1.0 - iou
                        if np.isnan(iou_cost).all():
                            mmt_to_gt_dist = 1.0
                        elif np.count_nonzero(~np.isnan(iou_cost)) / len(mmt_track) < min_overlap_ratio:
                            mmt_to_gt_dist = 1.0
                        else:
                            mmt_to_gt_dist = np.nanmean(iou_cost)

                    track_distances[mmt_idx, gt_idx] = mmt_to_gt_dist

            row_indices, col_indices = linear_sum_assignment(track_distances)
            track_distances_copy = track_distances.copy()
            track_distances_copy[row_indices, col_indices] = 1.0
            row_indices_secondary, col_indices_secondary = linear_sum_assignment(track_distances_copy)

            filtered_indices = np.where(track_distances[row_indices, col_indices] < max_iou_track_distance)
            filtered_row_indices = np.array(row_indices)[filtered_indices]
            filtered_col_indices = np.array(col_indices)[filtered_indices]
            mmt_gt_pairs = list(
                zip(model_detections.id.unique()[filtered_row_indices], ground_truth.id.unique()[filtered_col_indices])
            )

            filtered_indices_secondary = np.where(
                track_distances[row_indices_secondary, col_indices_secondary] < max_iou_track_distance
            )
            filtered_row_indices_secondary = np.array(row_indices_secondary)[filtered_indices_secondary]
            filtered_col_indices_secondary = np.array(col_indices_secondary)[filtered_indices_secondary]
            mmt_gt_pairs_secondary = list(
                zip(
                    model_detections.id.unique()[filtered_row_indices_secondary],
                    ground_truth.id.unique()[filtered_col_indices_secondary],
                )
            )

            all_mmt_gt_pairs.extend(mmt_gt_pairs)
            all_mmt_gt_pairs_secondary.extend(mmt_gt_pairs_secondary)

        return dict(all_mmt_gt_pairs), dict(all_mmt_gt_pairs_secondary)

    def plot_track_pairings(
        self, min_iou_thresh: float = 0.4, min_overlap_ratio: float = 0.3, max_iou_track_distance: float = 0.6
    ):
        if "cluster" not in self.mmt_dfs[0].columns:
            raise ValueError(
                "No clustering has been performed yet. Please perform clustering with the do_clustering method."
            )

        all_mmt_gt_pairs, all_mmt_gt_pairs_secondary = self.map_mmt2gt(
            min_iou_thresh, min_overlap_ratio, max_iou_track_distance
        )

        model_detections = pd.concat(self.mmt_dfs)
        ground_truth = pd.concat(self.gt_dfs)

        # Create a colormap with enough colors for each cluster
        n_clusters = model_detections["cluster"].nunique()
        colormap = cm.get_cmap("viridis", n_clusters + 1)  # +1 for the ground truth color

        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(10, 10))
        plt.gca().invert_yaxis()

        # Plot model detections
        for mmt_track_id, gt_track_id in all_mmt_gt_pairs.items():
            mmt_track_df = model_detections[model_detections.id == mmt_track_id]
            ax.plot(mmt_track_df.x, mmt_track_df.y, color=colormap(mmt_track_df.cluster.iloc[0] + 1))
            gt_track_df = ground_truth[ground_truth.id == gt_track_id]
            ax.plot(gt_track_df.x, gt_track_df.y, alpha=0.5, color=colormap(0), linestyle="dashed")

        for mmt_track_id, gt_track_id in all_mmt_gt_pairs_secondary.items():
            mmt_track_df = model_detections[model_detections.id == mmt_track_id]
            ax.plot(
                mmt_track_df.x, mmt_track_df.y, color=colormap(mmt_track_df.cluster.iloc[0] + 1), linestyle="dotted"
            )

        ax.set(ylabel="y", title="assigned trajectories with clustering", ylim=[270, 0], xlim=[0, 480])
        ax.set_aspect("equal", adjustable="box")

        # Create a custom legend
        legend_elements = [Line2D([0], [0], color=colormap(0), lw=2, label="Label")]
        for i in range(n_clusters):
            legend_elements.append(Line2D([0], [0], color=colormap(i + 1), lw=2, label=f"Cluster {i}"))
        ax.legend(handles=legend_elements)

        plt.show()

        # Plot tracks that were not assigned to a ground truth
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(10, 10))
        plt.gca().invert_yaxis()

        for mmt_track_id in model_detections.id.unique():
            if (mmt_track_id not in all_mmt_gt_pairs.keys()) and (
                mmt_track_id not in all_mmt_gt_pairs_secondary.keys()
            ):
                mmt_track_df = model_detections[model_detections.id == mmt_track_id]
                ax.plot(mmt_track_df.x, mmt_track_df.y, color=colormap(mmt_track_df.cluster.iloc[0] + 1))

        ax.set(ylabel="y", title="unassigned trajectories with clustering", ylim=[270, 0], xlim=[0, 480])
        ax.set_aspect("equal", adjustable="box")

        # Create a custom legend
        legend_elements = [Line2D([0], [0], color=colormap(0), lw=2, label="Label")]
        for i in range(n_clusters):
            legend_elements.append(Line2D([0], [0], color=colormap(i + 1), lw=2, label=f"Cluster {i}"))
        ax.legend(handles=legend_elements)

        plt.show()

    def do_clustering(self, features: list[str], clustering_method: Callable, n_clusters: int):
        labels = []
        for idx in range(len(self.mmt_dfs)):
            selected_features = pd.concat(self.mmt_dfs)[features]
            clustering = clustering_method(n_clusters=n_clusters)
            labels.append(clustering.fit_predict(selected_features))
            self.mmt_dfs[idx]["cluster"] = labels[-1]
        return labels
