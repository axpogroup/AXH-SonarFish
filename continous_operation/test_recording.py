import datetime as dt
import os
import subprocess

import yaml

if __name__ == "__main__":
    cwd = "/home/soundsedrun/code/AXH-Sound/acoustic_monitoring/"
    with open(os.path.join(cwd, "orchestrator_settings.yaml")) as f:
        settings_dict = yaml.load(f, Loader=yaml.SafeLoader)

    recording_directory = "sedrun_tests/recordings"
    os.makedirs(name=recording_directory, exist_ok=True)

    record_cmd_prefix = "ffmpeg -f alsa -i pulse -channels 2 -b:a 128"

    savepath = os.path.join(
        recording_directory,
        (
            "start_"
            + dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")
            + ".mp3"
        ),
    )
    savepath = savepath.replace(":", "-")
    print("Starting recording: ", savepath)
    recording_command = record_cmd_prefix + " " + savepath
    try:
        output = subprocess.run(
            recording_command,
            check=True,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(f"Saved: {savepath}")
    except subprocess.CalledProcessError as e:
        print(
            f"Error starting recording. \n Command: {recording_command} \n {e.output}"
        )
