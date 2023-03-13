import datetime as dt
import glob
import os

import utils
import yaml


def upload_latest_logs(n):
    log_files = glob.glob(
        os.path.join(settings_dict["recording_directory"], "*.mp3"), recursive=True
    )
    #        os.path.join(settings_dict["log_directory"], "**/*.log*"), recursive=True

    if len(log_files) == 0:
        raise Exception(
            f"No log files found in {os.path.join(settings_dict['log_directory'], '**/*.log*')}"
        )

    log_files = sorted(log_files, key=os.path.getmtime)
    # log_files = [os.path.join(cwd, "start_2023-03-02T10-58-20.819+00-00.mp3")]
    for log in log_files[-n:]:
        print("Attempting to upload Azure Storage as blob: " + log)
        cloud_handler.upload_file_to_container(log, "test")
        print("Success!")


if __name__ == "__main__":
    cwd = "/home/soundsedrun/code/AXH-Sound/acoustic_monitoring/"
    # cwd = os.getcwd()
    with open(os.path.join(cwd, "settings.yaml")) as f:
        settings_dict = yaml.load(f, Loader=yaml.SafeLoader)

    cloud_handler = utils.CloudHandler()
    print("Attempting to upload 10 logs...")
    upload_latest_logs(n=10)
    print("Sucess!")

    print("Attempting to send teams message...")
    cloud_handler.send_message(
        "green",
        "Test message.",
        f"Instance start time UTC: "
        f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='milliseconds')}",
    )
    print("Success!")
    print("All tests finished.")
