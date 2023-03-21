import argparse
import datetime as dt
import glob
import os
import subprocess
import sys
import time

sys.path.append("/home/fish-pi/code/")

import pandas as pd
import yaml
from utils import CloudHandler


def downsample_and_upload_recording(file):
    downsampled_rec_name = file[:-4] + "_downsampled.mp4"
    downsample_cmd = f"ffmpeg -y -i {file} -c:v libx264 -preset medium -crf 30 -vf scale=iw*0.25:ih*0.25 -r 10 -t 100 {downsampled_rec_name}"
    print(f"creating downsampled version with command: {downsample_cmd}")
    success = False
    try:
        output = subprocess.run(
            downsample_cmd,
            check=True,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(output.stdout)
        print("Snippet created: ", downsampled_rec_name)
        success = True
    except subprocess.CalledProcessError as e:
        print("-------- ERROR making snippet. ---------")
        print("Original file: " + file)
        print("Command: " + downsample_cmd)
        print("Output of subprocess: \n")
        print(e.output)

    time.sleep(3)

    # Upload the snippet
    if success:
        print(f"Attempting to upload downsampled recording: {downsampled_rec_name}")
        cloud_handler = CloudHandler()
        cloud_handler.upload_file_to_container(
            downsampled_rec_name, orchestrator_settings_dict["azure_container_name"]
        )
        print("Success!")


def upload_sample_of_latest_recording():
    existing_completed_recordings = pd.read_csv(
        os.path.join(
            orchestrator_settings_dict["file_list_directory"],
            "completed_recordings_list.csv",
        )
    )["path"].to_list()
    snippet_name = existing_completed_recordings[-1][:-4] + "_snippet.mp4"
    snippet_cmd = f"ffmpeg -y -i {existing_completed_recordings[-1]} -c:v libx264 -preset medium -crf 46 -t 00:00:10 {snippet_name}"
    print(f"creating snippet with command: {snippet_cmd}")
    success = False
    try:
        output = subprocess.run(
            snippet_cmd,
            check=True,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(output.stdout)
        print("Snippet created: ", snippet_name)
        success = True
    except subprocess.CalledProcessError as e:
        print("-------- ERROR making snippet. ---------")
        print("Original file: " + existing_completed_recordings[-1])
        print("Command: " + snippet_cmd)
        print("Output of subprocess: \n")
        print(e.output)

    time.sleep(3)

    # Upload the snippet
    if success:
        print(f"Attempting to upload snippet: {snippet_name}")
        cloud_handler = CloudHandler()
        cloud_handler.upload_file_to_container(
            snippet_name, orchestrator_settings_dict["azure_container_name"]
        )
        print("Success!")


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
    elif args.command == "start_recording_detection":
        pass
    elif args.command == "stop_all":
        pass
    elif args.command == "start_recording":
        pass
    elif args.command == "upload_logs":
        pass
    elif args.command == "upload_sample":
        upload_sample_of_latest_recording()
    elif args.command == "upload_file":
        print(f"Attempting to upload: {args.file}")
        cloud_handler = CloudHandler()
        cloud_handler.upload_file_to_container(
            args.file, orchestrator_settings_dict["azure_container_name"]
        )
        print("Success!")
    elif args.command == "downsample_upload":
        downsample_and_upload_recording(args.file)
    elif args.command == "feed_watchdog":
        # Write to watchdog
        pd.DataFrame(
            [dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")]
        ).to_csv(orchestrator_settings_dict["watchdog_food_file"])
    else:
        print(f"Command {args.command} not known.")
