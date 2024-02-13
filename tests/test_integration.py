import os

import pandas as pd
import pytest
import yaml

from algorithm.extract_labels_from_videos import main


@pytest.fixture(scope="session", autouse=True)
def setup():
    files = os.listdir("data/labels")
    assert (
        len(files) == 1 and files[0] == ".gitkeep"
    ), "The data/labels directory should be empty before running the test."
    yield
    files = os.listdir("data/labels")
    for file in files:
        if file != ".gitkeep":
            os.remove(f"data/labels/{file}")


class TestIntegration:

    def test_integration(self):
        with open("../settings/tracking_box_settings.yaml") as f:
            settings = yaml.load(f, Loader=yaml.SafeLoader)
        main(settings)

        labels_csv = pd.read_csv("data/labels/trimmed_video_ground_truth.csv")
        assert len(labels_csv) > 0
