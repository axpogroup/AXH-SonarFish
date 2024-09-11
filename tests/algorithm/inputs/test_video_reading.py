import sys
import unittest
from datetime import datetime, timezone

sys.path.append("..")
from algorithm.inputs.video_reading import (
    extract_timestamp_from_filename,
    extract_timestamp_from_lavey_fileformat,
    parse_date_from_stroppel_fileformat,
)


class TestVideoReading(unittest.TestCase):

    def test_extract_timestamp_from_filename(self):
        filenames = [
            (
                "start_2023-10-13T19-04-11.037+00-00.mp4",
                "start_%Y-%m-%dT%H-%M-%S.%f%z.mp4",
                datetime(2023, 10, 13, 19, 4, 11, 37000, tzinfo=timezone.utc),
            ),
            ("Passe3_Dec25_23-59-59.mp4", "sonarname_%b%d_%H-%M-%S.mp4", datetime(2024, 12, 25, 23, 59, 59)),
        ]
        for filename, format, expected in filenames:
            with self.subTest(filename=filename):
                self.assertEqual(extract_timestamp_from_filename(filename, format), expected)

    def test_extract_timestamp_from_lavey_fileformat(self):
        filenames = [
            ("Passe3_Apr08_07-00-00.mp4", datetime(2024, 4, 8, 7, 0, 0)),
            ("Passe3_May15_12-30-45.mp4", datetime(2024, 5, 15, 12, 30, 45)),
            ("Passe3_Dec25_23-59-59.mp4", datetime(2024, 12, 25, 23, 59, 59)),
        ]
        for filename, expected in filenames:
            with self.subTest(filename=filename):
                self.assertEqual(extract_timestamp_from_lavey_fileformat(filename), expected)

    def test_parse_date_from_stroppel_fileformat(self):
        filenames = [
            (
                "start_2023-10-13T19-04-11.037+00-00.mp4",
                datetime(2023, 10, 13, 19, 4, 11, 37000, tzinfo=timezone.utc),
            ),
            ("start_2024-01-01T00-00-00.000+00-00.mp4", datetime(2024, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)),
        ]
        for filename, expected in filenames:
            with self.subTest(filename=filename):
                self.assertEqual(parse_date_from_stroppel_fileformat(filename), expected)


if __name__ == "__main__":
    unittest.main()
