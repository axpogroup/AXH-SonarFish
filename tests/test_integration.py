import os

import pandas as pd
import pytest
import yaml

from algorithm.extract_labels_from_videos import main as label_extraction_main
from algorithm.preprocess_raw_videos import main as preprocess_main
from algorithm.run_algorithm import compute_metrics
from algorithm.run_algorithm import main as run_algorithm_main


@pytest.fixture(scope="session")
def labels_directory():
    return "data/labels"


@pytest.fixture(scope="session")
def intermediate_videos_directory():
    return "data/intermediate/videos"


@pytest.fixture(scope="session")
def intermediate_labels_directory():
    return "data/intermediate/labels"


@pytest.fixture(scope="session")
def model_output_directory():
    return "data/model_output"


@pytest.fixture(scope="session", autouse=True)
def setup(
    labels_directory,
    intermediate_videos_directory,
    intermediate_labels_directory,
    model_output_directory,
):
    assert_directory_empty(directory=labels_directory)
    assert_directory_empty(directory=intermediate_videos_directory)
    assert_directory_empty(directory=intermediate_labels_directory)
    assert_directory_empty(directory=model_output_directory)
    yield
    clear_directory(directory=labels_directory)
    clear_directory(directory=intermediate_videos_directory)
    clear_directory(directory=intermediate_labels_directory)
    clear_directory(directory=model_output_directory)


@pytest.fixture
def relevant_csv_columns():
    return ["frame", "id", "x", "y", "w", "h"]


def clear_directory(directory):
    files = os.listdir(directory)
    for file in files:
        if file != ".gitkeep":
            os.remove(f"{directory}/{file}")


def assert_directory_empty(directory: str):
    files = os.listdir(directory)
    if not (len(files) == 1 and files[0] == ".gitkeep"):
        raise Exception(f"The {directory} directory should be empty before running the test, but it has {files=}")


class TestIntegration:

    def test_integration(
        self,
        model_output_directory,
        intermediate_labels_directory,
        intermediate_videos_directory,
        labels_directory,
        relevant_csv_columns,
    ):
        with open("../settings/preprocessing_settings.yaml") as f:
            preprocess_settings = yaml.load(f, Loader=yaml.SafeLoader)

        preprocess_main(preprocess_settings)

        with open("../settings/tracking_box_settings.yaml") as f:
            label_extraction_settings = yaml.load(f, Loader=yaml.SafeLoader)
        label_extraction_main(label_extraction_settings)

        intermediate_labels = os.listdir(intermediate_labels_directory)
        intermediate_videos = os.listdir(intermediate_videos_directory)
        assert len(intermediate_labels) == 2
        assert len(intermediate_videos) == 2

        labels_csv = pd.read_csv(f"{labels_directory}/trimmed_video_ground_truth.csv")
        assert len(labels_csv) > 0
        assert list(labels_csv.columns)[:6] == relevant_csv_columns

        with open("../analysis/demo/demo_settings.yaml") as f:
            detection_settings = yaml.load(f, Loader=yaml.SafeLoader)
        detection_settings["file_name"] = "trimmed_video.mp4"
        detection_settings["mask_directory"] = "../analysis/demo/masks"
        detection_settings["display_output_video"] = False
        run_algorithm_main(detection_settings)
        detections_csv = pd.read_csv(f"{model_output_directory}/trimmed_video.csv")
        assert list(detections_csv.columns)[:6] == relevant_csv_columns
        assert len(detections_csv) > 0

        metrics = compute_metrics(detection_settings)
        assert len(metrics) > 0
