from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from algorithm.DetectedObject import BoundingBox
from algorithm.visualization_functions import TRUTH_LABEL_NO


def read_labels_into_dataframe(labels_path: Path, labels_filename: str) -> Optional[pd.DataFrame]:
    labels_path = Path(labels_path) / labels_filename
    if labels_path.exists():
        print(f"Found labels file: {labels_path}")
    else:
        print(f"No labels file found at: {labels_path}")
        return None
    return pd.read_csv(labels_path)


def extract_labels_history(
    label_history: dict[int, BoundingBox],
    labels: Optional[pd.DataFrame],
    current_frame: int,
    down_sample_factor: int = 1,
    feature_to_load: Optional[str] = None,
) -> Optional[dict[int, BoundingBox]]:
    if labels is None:
        return None
    # current_frame_df = labels[labels["frame"] == int(current_frame * down_sample_factor)]
    current_frame_df = labels[labels["frame"] == current_frame]
    for _, row in current_frame_df.iterrows():
        truth_detected = BoundingBox(
            identifier=row["id"],
            frame_number=row["frame"],
            contour=np.array(row[["x", "y", "w", "h"]]),
            label=int(row.get("assigned_label") or row.get("classification_v2", TRUTH_LABEL_NO)),
            precalculated_feature=row.get(feature_to_load, None),
        )
        if row["id"] not in label_history:
            label_history[row["id"]] = truth_detected
        else:
            label_history[row["id"]].update_object(truth_detected)
    return label_history
