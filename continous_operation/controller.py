import argparse
import datetime as dt
import glob
import os
import sys
sys.path.append("/home/fish-pi/code/")

import pandas as pd
import yaml

from utils import CloudHandler


def upload_sample_of_latest_recording():
    existing_completed_recordings = pd.read_csv(
        os.path.join(
            orchestrator_settings_dict["file_list_directory"],
            "completed_recordings_list.csv",
        )
    )["path"].to_list()

    print(f"Attempting to upload last recording: {existing_completed_recordings[-1]}")
    cloud_handler = CloudHandler()
    cloud_handler.upload_file_to_container(existing_completed_recordings[-1])


def modified_in_past_x_minutes(filepath, x):
    if (
        dt.datetime.now(dt.timezone.utc)
        - dt.datetime.fromtimestamp(os.path.getmtime(filepath), tz=dt.timezone.utc)
    ) < dt.timedelta(minutes=x):
        return True
    else:
        return False


def get_latest_logs():
    log_files = glob.glob(
        os.path.join(orchestrator_settings_dict["log_directory"], "**/*.log*"),
        recursive=True,
    )

    if len(log_files) == 0:
        raise Exception(
            f"No log files found in {os.path.join(orchestrator_settings_dict['log_directory'], '**/*.log*')}"
        )

    orchestrator_logs = [log for log in log_files if "orchestrator" in log]
    recording_logs = [log for log in log_files if "recording" in log]
    orchestrator_logs = sorted(orchestrator_logs, key=os.path.getmtime)
    recording_logs = sorted(recording_logs, key=os.path.getmtime)
    if len(recording_logs) == 0:
        print("no recording logs!")
        return orchestrator_logs[-1], None

    if len(orchestrator_logs) == 0:
        print("no orchestrator logs!")
        return None, recording_logs[-1]

    return orchestrator_logs[-1], recording_logs[-1]


def check_status():
    no_mod_thres = orchestrator_settings_dict[
        "error_after_no_file_modification_minutes"
    ]
    print("Checking recording status...")
    try:
        check_recordings()
        print("RECORDING RUNNING")
    except Exception as e:
        print("Recording not running: ", str(e))

    print("Getting latest log output...")
    orc, rec = get_latest_logs()
    if orc is not None:
        print("-------- Printing latest orchestrator output: --------", orc, "\n")
        os.system(f"tail -n 20 {orc}")
    if rec is not None:
        print("\n")
        print("-------- Printing latest recording output: --------", rec, "\n")
        os.system(f"tail -n 20 {rec}")


if __name__ == "__main__":
    argParser = argparse.ArgumentParser(
        description="Various controls for the continous fish detection system."
    )

    argParser.add_argument("-command", "--command", help="path to the input video file")
    argParser.add_argument("-file", "--file", help="path to the input video file")

    args = argParser.parse_args()

    cwd = "/home/fish-pi/code/continous_operation/"
    with open(os.path.join(cwd, "orchestrator_settings.yaml")) as f:
        orchestrator_settings_dict = yaml.load(f, Loader=yaml.SafeLoader)

    if args.command == "check_status":
        check_status()
    if args.command == "start_recording_detection":
        pass
    if args.command == "stop_all":
        pass
    if args.command == "start_recording":
        pass
    if args.command == "upload_logs":
        pass
    if args.command == "upload_sample":
        upload_sample_of_latest_recording()
    if args.command == "upload_file":
        cloud_handler = CloudHandler()
        cloud_handler.upload_file_to_container(yaml.file)
    if args.command == "feed_watchdog":
        # Write to watchdog
        pd.DataFrame(
            [dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")]
        ).to_csv(orchestrator_settings_dict["watchdog_food_file"])
