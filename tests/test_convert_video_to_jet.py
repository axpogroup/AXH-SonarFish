import os

import pytest

from algorithm.scripts.convert_video_red_to_jet import (
    process_video,  # replace with the actual module name
)


class TestProcessVideo:

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.input_filename = "./data/raw/videos/trimmed_video_minimal.mp4"
        self.output_filename = self.input_filename.rsplit(".", 1)[0] + "_jet.mp4"

        yield  # this is where the testing happens

        if os.path.exists(self.output_filename):
            os.remove(self.output_filename)

    def test_process_video(self):
        process_video(self.input_filename)

        # Add assertions here based on what the function is supposed to do
        # For example, if the function is supposed to create an output file:
        assert os.path.exists(self.output_filename)
