import argparse
import logging
import os
from pathlib import Path

import cv2 as cv
import mlflow
import yaml

from algorithm.run_algorithm import main_algorithm


def init():
    """Init."""
    global DATA_PATH
    global OUTPUT_PATH
    global TRACKING_CONFIG
    global LABELS_DIR
    global SAVE_OUTPUT_VIDEO
    global LOG_LEVEL

    parser = get_parser()
    args, _ = parser.parse_known_args()
    TRACKING_CONFIG = args.tracking_config
    if args.job_output_path is None:
        print("job_output_path is None, setting it to job_inputs_path/outputs")
        args.job_output_path = args.job_inputs_path + "/outputs"
    OUTPUT_PATH = args.job_output_path
    LOG_LEVEL = args.log_level
    DATA_PATH = args.job_inputs_path
    LABELS_DIR = args.labels_dir
    SAVE_OUTPUT_VIDEO = args.save_output_video
    print("OpenCV build information: ")
    print(cv.getBuildInformation())
    print("Pass through init done")


def run(mini_batch):
    """Run."""
    logging.basicConfig(level=logging.getLevelName(LOG_LEVEL))
    # logger = logging.getLogger(__name__)

    # mini_batch is a list of file paths for File Data
    for file_path in mini_batch:
        file = Path(file_path)
        file_base_path = file.parent.as_posix()
        file_name = file.name
        print("Processing {}".format(file))
        assert file.exists()
        assert file.suffix == ".mp4", "Only support .mp4 file"

        # Check if file size is over 50MB
        if os.stat(file_path).st_size < 50 * 1024 * 1024:  # size in bytes
            print(f"Skipping {file_name} as its size is less than 50MB.")
            continue

        # Two customers reported transient error when using OutputFileDatasetConfig.
        # It hits "FileNotFoundError" when writing to a file in the output_dir folder,
        #  even the folder did exist per logs.
        # This is to simulate such case and hope we can repro in our gated build.
        output_dir = Path(OUTPUT_PATH)
        print("output_dir", output_dir)
        print("output_dir exits", Path(output_dir).exists())
        # (Path(output_dir) / file.name).write_text(file_path)

        with open(TRACKING_CONFIG) as f:
            settings = yaml.load(f, Loader=yaml.SafeLoader)

        with mlflow.start_run():
            print(f"replacing input directory with {file_base_path}.")
            print(f"replacing output directory with {OUTPUT_PATH}.")
            print(f"replacing file name with {file_name}.")
            print(f"replacing ground truth directory with {LABELS_DIR}.")
            print(f"replacing record_output_video with {SAVE_OUTPUT_VIDEO}.")
            settings["input_directory"] = file_base_path
            settings["output_directory"] = OUTPUT_PATH
            settings["file_name"] = file_name
            settings["ground_truth_directory"] = LABELS_DIR or "."
            settings["record_output_video"] = SAVE_OUTPUT_VIDEO
            main_algorithm(settings)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fish detection algorithm with a settings .yaml file.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--job_inputs_path",
        type=str,
        help="path to the input video files directory or individual file path, \
              only needed when not running as parallel pipeline",
        required=True,
    )
    parser.add_argument(
        "--job_output_path",
        type=str,
        help="path to the output directory",
        default=None,
    )
    parser.add_argument(
        "--tracking_config",
        type=str,
        help="path to the YAML settings file",
        default="kalman_tracking_settings.yaml",
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        help="path to the directory containing the labels files, the correct one is chosen automatically",
        default=None,
    )
    parser.add_argument(
        "--save_output_video",
        type=str2bool,
        help="save the output video or not",
        default=False,
    )
    parser.add_argument(
        "--log_level",
        type=str,
        help="log level",
        default="INFO",
    )
    return parser


def main():
    if Path(DATA_PATH).is_dir():
        input_video_file_paths = list(Path(DATA_PATH).glob("*.mp4"))
        print(f"Found {len(input_video_file_paths)} video files in the given directory")
    else:
        input_video_file_paths = [DATA_PATH]
    print(f"Processing {input_video_file_paths}")
    run(input_video_file_paths)


if __name__ == "__main__":
    init()
    main()
