import os

import pandas as pd
import pytest
import yaml

from algorithm.extract_labels_from_videos import main as label_extraction_main
from algorithm.preprocess_raw_videos import main as preprocess_main


@pytest.fixture(scope="session", autouse=True)
def setup():
    assert_directory_empty(directory="data/labels")
    assert_directory_empty(directory="data/intermediate/videos")
    assert_directory_empty(directory="data/intermediate/labels")
    yield
    clear_directory(directory="data/labels")
    clear_directory(directory="data/intermediate/videos")
    clear_directory(directory="data/intermediate/labels")


def clear_directory(directory):
    files = os.listdir(directory)
    for file in files:
        if file != ".gitkeep":
            os.remove(f"{directory}/{file}")


def assert_directory_empty(directory: str):
    files = os.listdir(directory)
    assert (
        len(files) == 1 and files[0] == ".gitkeep"
    ), f"The {directory} directory should be empty before running the test."


class TestIntegration:

    def test_integration(self):
        with open("../settings/preprocessing_settings.yaml") as f:
            preprocess_settings = yaml.load(f, Loader=yaml.SafeLoader)

        preprocess_main(preprocess_settings)

        with open("../settings/tracking_box_settings.yaml") as f:
            label_extraction_settings = yaml.load(f, Loader=yaml.SafeLoader)
        label_extraction_main(label_extraction_settings)

        intermediate_labels = os.listdir("data/intermediate/labels")
        intermediate_videos = os.listdir("data/intermediate/videos")
        assert len(intermediate_labels) == 2
        assert len(intermediate_videos) == 2

        labels_csv = pd.read_csv("data/labels/trimmed_video_ground_truth.csv")
        assert len(labels_csv) > 0
