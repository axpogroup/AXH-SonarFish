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
    recording_uploads_dir = os.path.join(
        os.path.split(orchestrator_settings_dict["detections_directory"])[0],
        "recording_uploads",
    )
    os.makedirs(recording_uploads_dir, exist_ok=True)
    downsampled_rec_name = os.path.join(
        recording_uploads_dir, (os.path.split(file)[-1][:-4] + "_downsampled.mp4")
    )
    downsample_cmd = (
        f"ffmpeg -y -i {file} -c:v libx264 -preset medium -crf 30 -vf scale=iw*0.25:ih*0.25 -r 10 "
        f"{downsampled_rec_name}"
    )
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
    recording_uploads_dir = os.path.join(
        os.path.split(orchestrator_settings_dict["detections_directory"])[0],
        "recording_uploads",
    )
    os.makedirs(recording_uploads_dir, exist_ok=True)

    existing_completed_recordings = pd.read_csv(
        os.path.join(
            orchestrator_settings_dict["file_list_directory"],
            "completed_recordings_list.csv",
        )
    )["path"].to_list()
    snippet_name = os.path.join(
        recording_uploads_dir,
        (os.path.split(existing_completed_recordings[-1])[-1][:-4] + "_snippet.mp4"),
    )
    snippet_cmd = (
        f"ffmpeg -y -i {existing_completed_recordings[-1]} -c:v libx264 -preset medium -crf 46 -t 00:00:10 "
        f"{snippet_name}"
    )
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

    return orchestrator_logs, recording_logs


def check_status():
    def check_recordings():
        all_recordings = glob.glob(
            os.path.join(orchestrator_settings_dict["recording_directory"], "*.mp4")
        )
        if len(all_recordings) == 0:
            raise Exception("No recordings found.")

        sorted(all_recordings, key=os.path.getmtime)
        if not modified_in_past_x_minutes(all_recordings[-1], no_mod_thres):
            raise Exception(f"No file modification in the past {no_mod_thres} minutes.")

    no_mod_thres = orchestrator_settings_dict[
        "error_after_no_file_modification_minutes"
    ]
    print("Checking recording status...")
    try:
        check_recordings()
        print("\n")
        print("RECORDING RUNNING")
        print("\n")
    except Exception as e:
        print("\n")
        print("Recording not running: ", str(e))

    print("Getting latest log output...")
    orc, rec = get_latest_logs()
    if len(rec) == 0:
        print("no recording logs!")
    else:
        print("-------- Printing latest recording output: --------", rec[-1], "\n")
        os.system(f"tail -n 20 {rec[-1]}")
    if len(orc) == 0:
        print("no orchestrator logs!")
    else:
        print("\n")
        print("-------- Printing latest orchestrator output: --------", orc[-1], "\n")
        os.system(f"tail -n 20 {orc[-1]}")


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
    # elif args.command == "start_recording_detection":
    #     pass
    # elif args.command == "stop_all":
    #     pass
    # elif args.command == "start_recording":
    #     pass
    elif args.command == "upload_logs":
        print("Getting latest logs ...")
        orc_logs, rec_logs = get_latest_logs()
        cloud_handler = CloudHandler()
        print("Uploading latest 5 orchestrator and recording logs...")
        if len(orc_logs) > 5:
            orc_logs = orc_logs[-5]
        if len(rec_logs) > 5:
            rec_logs = rec_logs[-5]
        upload = orc_logs + rec_logs
        for file in upload:
            cloud_handler.upload_file_to_container(
                file, orchestrator_settings_dict["azure_container_name"]
            )
        print("Success!")

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
        pd.DataFrame(
            [dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")]
        ).to_csv(orchestrator_settings_dict["watchdog_food_file"])
        print("Fed watchdog.")
    elif len(args.command) == 0:
        print("Use one of the following commands:")
        options = [
            "check_status",
            "upload_logs",
            "upload_sample",
            "upload_file",
            "downsample_upload",
            "feed_watchdog",
        ]
        for option in options:
            print(option)
    else:
        print(f"Command {args.command} not known.")
