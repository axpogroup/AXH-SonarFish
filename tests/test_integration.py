import os
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from algorithm.run_algorithm import compute_metrics
from algorithm.run_algorithm import main_algorithm as run_algorithm_main
from algorithm.scripts.extract_labels_from_videos import main as label_extraction_main
from algorithm.scripts.preprocess_raw_videos import main as preprocess_main
from algorithm.settings import Settings


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
        if file != ".gitkeep" and not (directory.endswith("raw/labels") and file == "trimmed_video.mp4"):
            os.remove(f"{directory}/{file}")


def assert_directory_empty(directory: str):
    allowed_files = [
        ".gitkeep",
        "trimmed_video.mp4",
        "trimmed_video_minimal.mp4",
        "start_2023-05-08T18-00-05.025+00-00_ground_truth.csv",
        "start_2023-05-08T18-00-05.025+00-00.mp4",
    ]
    files = os.listdir(directory)
    disallowed_files = [file for file in files if file not in allowed_files]
    if disallowed_files:
        raise Exception(
            f"The {directory} directory should be empty before running the test, but it has {disallowed_files=}"
        )


class TestIntegration(unittest.TestCase):

    @patch("argparse.ArgumentParser.parse_args")
    def test_integration(
        self,
        mock_parse_args,
        model_output_directory,
        intermediate_labels_directory,
        intermediate_videos_directory,
        labels_directory,
        relevant_csv_columns,
    ):
        # Mock the arguments
        mock_parse_args.return_value = MagicMock(yaml_file="../settings/demo_settings.yaml", input_file=None)

        with open("../settings/demo_settings.yaml") as f:
            detection_settings = yaml.load(f, Loader=yaml.SafeLoader)

        detection_settings = Settings(**detection_settings)
        run_algorithm_main(detection_settings)
        detections_csv = pd.read_csv(f"{model_output_directory}/start_2023-05-08T18-00-05.025+00-00.csv")
        assert list(detections_csv.columns)[:6] == relevant_csv_columns
        assert len(detections_csv) > 0

        output_files = os.listdir(model_output_directory)
        assert len(output_files) == 3  # video_compressed, csv, .gitkeep

        metrics = compute_metrics(detection_settings)
        assert len(metrics) > 0
