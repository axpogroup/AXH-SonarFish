import datetime as dt
import glob
import os
import subprocess

import numpy as np
from dateutil.relativedelta import relativedelta

if __name__ == "__main__":
    # This script changes the filenames since the raspberry pi had the wrong system time
    input_files_stick = "/Volumes/SONAR_STICK/recordings_stroppel_weekend/*.mp4"
    input_files_local = (
        "/Users/leivandresen/Documents/PROJECTS/SONAR_FISH/Field_test_Stroppel_20to24_10_22/"
        "weekend_backup/*.mp4"
    )
    input_files_test = (
        "/Users/leivandresen/Documents/Hydro_code/AXH-SonarFish/file_stitch_test/*.mp4"
    )
    save_dir = "/Users/leivandresen/Documents/Hydro_code/AXH-SonarFish/file_stitch_test/stitched/"
    os.makedirs(name=save_dir, exist_ok=True)
    n_videos_to_join = 2

    filenames = glob.glob(input_files_test)
    filenames.sort()

    current_start_file = 0
    while current_start_file < len(filenames):
        interest = filenames[current_start_file:]
        if len(filenames[current_start_file:]) > n_videos_to_join:
            interest = filenames[
                current_start_file: (current_start_file + n_videos_to_join)
            ]

        # Sample "[0:0][1:0][2:0][3:0][4:0][5:0]concat=n=6:v=1:a=0[v]"
        input_files = "".join([(" -i " + file) for file in interest])
        input_params = (
            "'"
            + "".join([f"[{file_no}:0]" for file_no in np.arange(0, len(interest))])
            + f"concat=n={len(interest)}:v=1:a=0[v]"
            + "'"
        )

        date_fmt = "%y-%m-%d_start_%H-%M-%S.mp4"
        datetime_start = dt.datetime.strptime(os.path.split(interest[0])[-1], date_fmt)
        datetime_end = dt.datetime.strptime(
            os.path.split(interest[-1])[-1], date_fmt
        ) + relativedelta(minutes=10)

        new_filename = (
            save_dir
            + datetime_start.strftime("%y-%m-%d_start_%H-%M-%S")
            + datetime_end.strftime("_end_%H-%M-%S")
            + ".mp4"
        )

        command = (
            "ffmpeg "
            + input_files
            + " -filter_complex "
            + input_params
            + " -map '[v]' -c:v libx264 -preset medium -crf 40 "
            + new_filename
        )

        print(command)
        try:
            output = subprocess.run(
                command,
                check=True,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            print(output.stdout)
            print(
                "\n\n"
                + dt.datetime.now().strftime("%y-%m-%d__%H-%M-%S")
                + " -------- processed: "
                + new_filename
                + "\n\n"
            )
        except subprocess.CalledProcessError as e:
            print(
                "\n\n"
                + dt.datetime.now().strftime("%y-%m-%d__%H-%M-%S")
                + " -------- ERROR: \n"
            )
            print("Savepath: " + new_filename)
            print("Command: " + command)
            print("Output of subprocess: \n")
            print(e.output)
            print("\nSleeping 2 seconds...\n")
            quit()

        current_start_file += len(interest)
