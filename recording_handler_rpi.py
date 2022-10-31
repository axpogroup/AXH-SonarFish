import os
import subprocess
import time
from datetime import datetime as dt

if __name__ == "__main__":
    # directory = "recordings_10min_Stroppel/"
    directory = "/media/fish-pi/SONAR_STICK/recordings_stroppel_weekend/"
    date_fmt = "%y-%m-%d_start_%H-%M-%S"
    record_cmd_prefix = "ffmpeg -framerate 25 -pixel_format uyvy422 -i /dev/video0 -vcodec h264_v4l2m2m -b:v 6M -r 20"
    duration_suffix = "-t 00:10:00"

    os.makedirs(name=directory, exist_ok=True)

    fatal_error = False
    while not fatal_error:
        savepath = directory + dt.now().strftime(date_fmt) + ".mp4"
        recording_command = record_cmd_prefix + " " + duration_suffix + " " + savepath
        try:
            output = subprocess.run(
                recording_command,
                check=True,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            print(output.stdout)
            print(
                "\n\n"
                + dt.now().strftime("%y-%m-%d__%H-%M-%S")
                + " -------- saved: "
                + savepath
                + "\n\n"
            )
        except subprocess.CalledProcessError as e:
            print(
                "\n\n"
                + dt.now().strftime("%y-%m-%d__%H-%M-%S")
                + " -------- ERROR starting recording. \n"
            )
            print("Savepath: " + savepath)
            print("Command: " + recording_command)
            print("Output of subprocess: \n")
            print(e.output)
            print("\nSleeping 2 seconds...\n")
            time.sleep(2)
