import datetime
import glob
import os

from dateutil.relativedelta import relativedelta

if __name__ == "__main__":
    # This script changes the filenames since the raspberry pi had the wrong system time
    files_stick = "/Volumes/SONAR_STICK/recordings_stroppel_weekend/*.mp4"
    files_local = (
        "/Users/leivandresen/Documents/PROJECTS/SONAR_FISH/Field_test_Stroppel_20to24_10_22/"
        "weekend_backup/*.mp4"
    )
    files_test = "/Users/leivandresen/Documents/Hydro_code/AXH-SonarFish/old_recording_handler_tests/*.mp4"

    offset = relativedelta(hours=4, minutes=48, seconds=46)

    filenames = glob.glob(files_stick)

    for file in filenames:

        name = os.path.split(file)[-1]
        time = name[-12:-4]
        date = name[:8]
        datetime_rpi = datetime.strptime(name[:-4], "%y-%m-%d_start_%H-%M-%S")
        datetime_real = datetime_rpi + offset

        new_filename = (
            os.path.split(file)[0]
            + "/"
            + datetime_real.strftime("%y-%m-%d_start_%H-%M-%S")
            + ".mp4"
        )

        # os.rename(file, new_filename)
