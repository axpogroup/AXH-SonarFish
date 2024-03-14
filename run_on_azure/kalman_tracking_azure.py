import argparse
import logging
from pathlib import Path

import mlflow
import yaml

from algorithm.run_algorithm import main_algorithm


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the fish detection algorithm with a settings .yaml file.")
    parser.add_argument("--tracking_config", type=str, help="path to the YAML settings file", required=True)
    parser.add_argument("--dataset", type=str, help="path to the input video files directory", required=True)
    parser.add_argument("--output", type=str, help="path to the output directory", required=True)
    parser.add_argument("--log_level", type=int, help="log level", default=logging.INFO)
    return parser


def main(args: argparse.Namespace):
    DATA_PATH = args.dataset
    DATA_OUTPUT_PATH = args.output

    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger(__name__)

    with open(args.tracking_config) as f:
        settings = yaml.load(f, Loader=yaml.SafeLoader)
        if DATA_OUTPUT_PATH is not None:
            logger.info(f"replacing output directory with {DATA_OUTPUT_PATH}.")
            settings["output_directory"] = DATA_OUTPUT_PATH

    # Training Step
    with mlflow.start_run():
        input_video_file_paths = Path(DATA_PATH).glob("**/*.mp4")
        logger.info(f"Found {len(list(input_video_file_paths))} video files in {DATA_PATH}")
        for input_video_path in input_video_file_paths:
            settings["file_name"] = input_video_path.name
            main_algorithm(settings)


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
