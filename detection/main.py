import argparse

import yaml
from FishDetector import FishDetector
from InputOutputHandler import InputOutputHandler

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-yf", "--yaml_file", help="YAML settings file")

    args = argParser.parse_args()

    with open(args.yaml_file) as f:
        settings_dict = yaml.load(f, Loader=yaml.SafeLoader)

    input_output_handler = InputOutputHandler(settings_dict)
    detector = FishDetector(settings_dict)

    while input_output_handler.get_new_frame():
        detector.process_frame(input_output_handler.current_raw_frame)
        input_output_handler.handle_output(detector)

    del detector
