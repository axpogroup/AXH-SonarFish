import datetime as dt
import os
import subprocess
import time

import utils
import yaml

if __name__ == "__main__":
    cwd = "/home/soundsedrun/code/AXH-Sound/acoustic_monitoring/"
    with open(os.path.join(cwd, "orchestrator_settings.yaml")) as f:
        settings_dict = yaml.load(f, Loader=yaml.SafeLoader)

    logger = utils.get_logger(settings_dict["log_directory"], "recording")
    os.makedirs(name=settings_dict["recording_directory"], exist_ok=True)

    duration = int(settings_dict["recording_interval_minutes"] * 60)
    record_cmd_prefix = f"ffmpeg -f alsa -i pulse -t {duration} -channels 2 -b:a 128"

    logger.info("Starting recording...")
    while True:
        savepath = os.path.join(
            settings_dict["recording_directory"],
            (
                "start_"
                + dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")
                + ".mp3"
            ),
        )
        savepath = savepath.replace(":", "-")
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
            logger.info(f"Saved: {savepath}")
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Error starting recording. \n Command: {recording_command} \n {e.output}"
            )
            time.sleep(30)
