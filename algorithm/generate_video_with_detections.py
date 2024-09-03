import argparse
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append(".")
from algorithm.InputOutputHandler import InputOutputHandler
from algorithm.run_algorithm import extract_labels_history, read_labels_into_dataframe
from algorithm.settings import Settings, settings

load_dotenv()


class DummyFishDetector:

    def __init__(self, local_settings: Settings):
        self.__settings = local_settings
        self.__settings.show_detections = False
        self.__settings.downsample = 25
        self.__settings.verbosity = 1
        self.frame_number = 0


def main_draw_annotations():
    labels_df = read_labels_into_dataframe(
        labels_path=settings.ground_truth_directory,
        labels_filename=settings.file_name.stem + settings.labels_file_suffix + settings.ground_truth + ".csv",
    )

    input_output_handler = InputOutputHandler()
    detector = DummyFishDetector()
    label_history = {}
    print("Starting annotating video.")

    with tqdm(total=input_output_handler.frames_total, desc="Processing frames") as pbar:
        while input_output_handler.get_new_frame():
            detector.frame_number += 1
            frame_dict = {"raw": input_output_handler.current_raw_frame}
            label_history = extract_labels_history(
                label_history,
                labels_df,
                input_output_handler.frame_no,
                down_sample_factor=input_output_handler.down_sample_factor,
                feature_to_load=self.__settings.feature_to_load,
            )
            input_output_handler.handle_output(
                processed_frame=frame_dict,
                object_history={},
                label_history=label_history,
                runtimes=None,
                detector=detector,
            )
            pbar.update(1)

    if settings.record_output_video and settings.compress_output_video:
        input_output_handler.compress_output_video()
        input_output_handler.delete_temp_output_dir()

    return None


if __name__ == "__main__":
    argParser = argparse.ArgumentParser(description="Run the fish detection algorithm with a settings .yaml file.")
    argParser.add_argument("-yf", "--yaml_file", help="path to the YAML settings file", required=True)
    argParser.add_argument("-if", "--input_file", help="path to the input video file")

    args = argParser.parse_args()

    with open(args.yaml_file) as f:
        settings = yaml.load(f, Loader=yaml.SafeLoader)
        if args.input_file is not None:
            print("replacing input file.")
            settings["file_name"] = args.input_file

    main_draw_annotations(settings)
