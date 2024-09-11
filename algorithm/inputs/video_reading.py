import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


def extract_timestamp_from_filename(
    filename: Union[str, Path],
    file_timestamp_format: str,
) -> Optional[datetime]:
    if isinstance(filename, Path):
        filename = filename.name
    if file_timestamp_format == "start_%Y-%m-%dT%H-%M-%S.%f%z.mp4":
        return parse_date_from_stroppel_fileformat(filename)
    elif file_timestamp_format == "sonarname_%b%d_%H-%M-%S.mp4":
        return extract_timestamp_from_lavey_fileformat(filename)
    else:
        raise ValueError(f"Unknown file timestamp format: {file_timestamp_format}")


def extract_timestamp_from_lavey_fileformat(
    filename: str,
    year: int = 2024,
) -> Optional[datetime]:
    date_str = filename.split("_")[1] + "_" + filename.split("_")[2].split(".")[0]
    parsed_date = datetime.strptime(date_str, "%b%d_%H-%M-%S")
    final_date = parsed_date.replace(year=year)
    return final_date


def parse_date_from_stroppel_fileformat(
    filename: str,
) -> datetime:
    date_str = filename.split("_")[1].rsplit(".", 1)[0]

    # Convert the time zone offset to the correct format
    date_str = re.sub(r"([+-]\d{2})[+-](\d{2})$", r"\1:\2", date_str)

    return datetime.strptime(date_str, "%Y-%m-%dT%H-%M-%S.%f%z")
