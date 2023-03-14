import datetime as dt
import os
import sys
sys.path.append("/home/fish-pi/code/")
import subprocess
import time

from continous_operation import utils
import yaml

if __name__ == "__main__":
    capture_initialization_command = "sh /home/fish-pi/code/continous_operation/initialize_capture/initialize_capture.sh"
    logger.info("Initializing capture device ...")
    try:
        output = subprocess.run(
            capture_initialization_command,
            check=True,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        logger.info(output.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(
            f"Error. \n {e.output}"
        )

    time.sleep(5)

    cwd = "/home/fish-pi/code/continous_operation/"
    with open(os.path.join(cwd, "orchestrator_settings.yaml")) as f:
        settings_dict = yaml.load(f, Loader=yaml.SafeLoader)

    logger = utils.get_logger(settings_dict["log_directory"], "recording")
    os.makedirs(name=settings_dict["recording_directory"], exist_ok=True)

    duration = int(settings_dict["recording_interval_minutes"] * 60)
    record_cmd_prefix = f"ffmpeg -framerate 25 -pixel_format uyvy422 -i /dev/video0 -vcodec h264_v4l2m2m -b:v 6M -r 20 -t {duration}"

    logger.info("Starting recording...")
    while True:
        savepath = os.path.join(
            settings_dict["recording_directory"],
            (
                "start_"
                + dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")
                + ".mp4"
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
