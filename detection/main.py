import argparse

import yaml
from FishDetector import FishDetector
from InputOutputHandler import InputOutputHandler

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-yf", "--yaml_file", help="YAML settings file")
    argParser.add_argument("-if", "--input_file", help="input_video_file")

    args = argParser.parse_args()

    with open(args.yaml_file) as f:
        settings_dict = yaml.load(f, Loader=yaml.SafeLoader)
        if args.input_file is not None:
            print("replacing input file.")
            settings_dict["input_file"] = args.input_file

    input_output_handler = InputOutputHandler(settings_dict)
    detector = FishDetector(settings_dict)

    while input_output_handler.get_new_frame():
        detector.process_frame(input_output_handler.current_raw_frame)
        input_output_handler.handle_output(detector)

    del detector
    del input_output_handler
