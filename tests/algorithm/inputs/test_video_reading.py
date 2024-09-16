import sys
from datetime import datetime, timezone

import pytest

sys.path.append("..")
from algorithm.inputs.video_reading import (
    extract_timestamp_from_filename,
    extract_timestamp_from_lavey_fileformat,
    parse_date_from_stroppel_fileformat,
)


@pytest.mark.parametrize(
    "filename, format, expected",
    [
        (
            "start_2023-10-13T19-04-11.037+00-00.mp4",
            "start_%Y-%m-%dT%H-%M-%S.%f%z.mp4",
            datetime(2023, 10, 13, 19, 4, 11, 37000, tzinfo=timezone.utc),
        ),
        (
            "Passe3_Dec25_23-59-59.mp4",
            "sonarname_%b%d_%H-%M-%S.mp4",
            datetime(2024, 12, 25, 23, 59, 59),
        ),
    ],
)
def test_extract_timestamp_from_filename(filename, format, expected):
    assert extract_timestamp_from_filename(filename, format) == expected


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("Passe3_Apr08_07-00-00.mp4", datetime(2024, 4, 8, 7, 0, 0)),
        ("Passe3_May15_12-30-45.mp4", datetime(2024, 5, 15, 12, 30, 45)),
        ("Passe3_Dec25_23-59-59.mp4", datetime(2024, 12, 25, 23, 59, 59)),
    ],
)
def test_extract_timestamp_from_lavey_fileformat(filename, expected):
    assert extract_timestamp_from_lavey_fileformat(filename) == expected


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("start_2023-10-13T19-04-11.037+00-00.mp4", datetime(2023, 10, 13, 19, 4, 11, 37000, tzinfo=timezone.utc)),
        ("start_2024-01-01T00-00-00.000+00-00.mp4", datetime(2024, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)),
    ],
)
def test_parse_date_from_stroppel_fileformat(filename, expected):
    assert parse_date_from_stroppel_fileformat(filename) == expected


if __name__ == "__main__":
    pytest.main()
