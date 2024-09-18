import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union

import cv2 as cv


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

    # Convert the time zone offset to the correct format (i.e. +02+00 -> +02:00)
    date_str = re.sub(r"([+-]\d{2})[+-](\d{2})$", r"\1:\2", date_str)

    return datetime.strptime(date_str, "%Y-%m-%dT%H-%M-%S.%f%z")


def get_video_duration(filepath: Path):
    video = cv.VideoCapture(str(filepath))
    frames = video.get(cv.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv.CAP_PROP_FPS)
    duration = frames / fps
    video.release()
    return duration


def find_valid_previous_video(settings_dict, gap_seconds: int):
    print("Finding a valid previous video...")
    current_timestamp = extract_timestamp_from_filename(
        settings_dict["file_name"], settings_dict["file_timestamp_format"]
    )

    # video files with the same starting letters as the current video
    video_files = sorted(Path(settings_dict["input_directory"]).glob(f"{settings_dict['file_name'][:4]}*.mp4"))
    closest_video = None

    min_time_difference = None
    for video_file in video_files:
        try:
            video_timestamp = extract_timestamp_from_filename(video_file.name, settings_dict["file_timestamp_format"])
        except ValueError:
            continue
        time_difference = current_timestamp - video_timestamp
        if time_difference.total_seconds() <= 0:
            continue
        elif min_time_difference is None or abs(time_difference) < abs(min_time_difference):
            closest_video = video_file
            min_time_difference = time_difference

    # Check the duration of the closest video to make sure there is no gap
    if closest_video:
        duration = get_video_duration(closest_video)
        end_timestamp = extract_timestamp_from_filename(
            closest_video.name, settings_dict["file_timestamp_format"]
        ) + timedelta(seconds=duration)
        if abs(current_timestamp - end_timestamp) <= timedelta(seconds=gap_seconds):
            print(
                print(
                    f"Closest video found: {closest_video.name}, "
                    f"time gap between recordings: {(current_timestamp - end_timestamp).total_seconds()} seconds."
                )
            )
            return closest_video.name
        else:
            print(
                f"Closest video {closest_video.name} ends {(current_timestamp - end_timestamp)} "
                f"before the current video, which is outside the tolerance ({gap_seconds} seconds)."
            )
            return None
    else:
        print("No previous video found.")
        return None
