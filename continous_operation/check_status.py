import datetime as dt
import glob
import os

import yaml


def modified_in_past_x_minutes(filepath, x):
    if (
        dt.datetime.now(dt.timezone.utc)
        - dt.datetime.fromtimestamp(os.path.getmtime(filepath), tz=dt.timezone.utc)
    ) < dt.timedelta(minutes=x):
        return True
    else:
        return False


def check_recordings():
    all_recordings = glob.glob(
        os.path.join(settings_dict["recording_directory"], "*.mp4")
    )
    if len(all_recordings) == 0:
        raise Exception("No recordings found.")

    sorted(all_recordings, key=os.path.getmtime)
    if not modified_in_past_x_minutes(all_recordings[-1], no_mod_thres):
        raise Exception(f"No file modification in the past {no_mod_thres} minutes.")


def get_latest_logs():
    log_files = glob.glob(
        os.path.join(settings_dict["log_directory"], "**/*.log*"), recursive=True
    )

    if len(log_files) == 0:
        raise Exception(
            f"No log files found in {os.path.join(settings_dict['log_directory'], '**/*.log*')}"
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


if __name__ == "__main__":
    cwd = "/home/fish-pi/code/continous_operation/"
    with open(os.path.join(cwd, "orchestrator_settings.yaml")) as f:
        settings_dict = yaml.load(f, Loader=yaml.SafeLoader)

    no_mod_thres = settings_dict["error_after_no_file_modification_minutes"]
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
