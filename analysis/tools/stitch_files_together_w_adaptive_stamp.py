# flake8: noqa

import datetime as dt
import glob
import os
import subprocess

from dateutil.relativedelta import relativedelta


#  ---- CHECKING TIME DIFFS BETWEEN VIDEOS ----
def check_time_gaps(filename_list, date_format):
    time_diffs = []
    for i in range(1, len(filename_list)):
        datetime_start_2 = dt.datetime.strptime(os.path.split(filename_list[i])[-1], date_format)
        datetime_end_1 = dt.datetime.strptime(os.path.split(filename_list[i - 1])[-1], date_format) + relativedelta(
            minutes=10
        )
        delta_s = (datetime_start_2 - datetime_end_1).total_seconds()
        time_diffs.append(delta_s)

    # time_diffs.sort()
    # print("Biggest time gaps: ", time_diffs[-10:])
    return time_diffs


if __name__ == "__main__":
    input_files_stick = "/Volumes/SONAR_STICK/Stroppel_ongoing/*.mp4"
    input_files_disk = "/Volumes/sonar-disk/Stroppel_ongoing/*.mp4"

    input_files_local = (
        "/Users/leivandresen/Documents/PROJECTS/SONAR_FISH/Field_test_Stroppel_20to24_10_22/" "weekend_backup/*.mp4"
    )
    input_files_test = "/Users/leivandresen/Documents/Hydro_code/AXH-SonarFish/file_stitch_test/*.mp4"
    # save_dir = "/Volumes/SONAR_STICK/recordings_stroppel_ongoing_4h_compressed/"
    save_dir = "/Volumes/sonar-disk/Aufnahmen_02bis07_11_22_4h_komprimiert_new/"

    os.makedirs(name=save_dir, exist_ok=True)
    date_fmt = "%y-%m-%d_start_%H-%M-%S.mp4"
    n_videos_to_join = 24

    filenames = glob.glob(input_files_disk)
    filenames.sort()

    set_start_date = True
    set_end_date = True
    start_date = dt.datetime.strptime("22-11-06_start_02-46-00.mp4", date_fmt)
    end_date = dt.datetime.strptime("22-11-07_start_14-27-00.mp4", date_fmt)
    if set_start_date:
        for i in range(0, len(filenames)):
            datetime_file = dt.datetime.strptime(os.path.split(filenames[i])[-1], date_fmt)
            if datetime_file >= start_date:
                filenames = filenames[i:]
                break

    if set_end_date:
        for i in range(0, len(filenames)):
            datetime_file = dt.datetime.strptime(os.path.split(filenames[i])[-1], date_fmt)
            if datetime_file > end_date:
                filenames = filenames[:i]
                break

    current_start_file = 0
    epoch = dt.datetime.utcfromtimestamp(0)
    while current_start_file < len(filenames):
        interest = filenames[current_start_file:]
        if len(interest) > n_videos_to_join:
            interest = filenames[current_start_file : (current_start_file + n_videos_to_join)]

        # Check for time differences bigger than 15 seconds
        time_between_videos = check_time_gaps(interest, date_fmt)
        for index, gap in enumerate(time_between_videos):
            if gap > 120:
                print("Happened: ", interest[index], " and ", interest[index + 1])
                interest = interest[: index + 1]
                time_between_videos = check_time_gaps(interest, date_fmt)
                print(time_between_videos, interest)
                break

        datetime_start = dt.datetime.strptime(os.path.split(interest[0])[-1], date_fmt)
        datetime_end = dt.datetime.strptime(os.path.split(interest[-1])[-1], date_fmt) + relativedelta(minutes=10)
        seconds_since_epoch_start = (datetime_start - epoch).total_seconds()

        new_filename = (
            save_dir
            + datetime_start.strftime("%y-%m-%d_start_%H-%M")
            + datetime_end.strftime("_end_%H-%M_compressed")
            + ".mp4"
        )

        input_files = "".join([(" -i " + file) for file in interest])
        filter_complex = ""
        final_streams = ""
        for file_no, filename in enumerate(interest):
            datetime_start = dt.datetime.strptime(os.path.split(filename)[-1], date_fmt)
            seconds_since_epoch_start = (datetime_start - epoch).total_seconds()
            filter_complex += (
                f"[{file_no}:0]drawtext=text='%{{pts\:gmtime\:{seconds_since_epoch_start}}}'"
                f":x=(w-550):y=(h-130):fontfile=OpenSans.ttf:fontsize=40:fontcolor=white[{file_no}v];"
            )
            final_streams += f"[{file_no}v]"

        filter_complex = (
            " -filter_complex " + '"' + filter_complex + final_streams + f"concat=n={len(interest)}:v=1:a=0[v]" + '"'
        )

        command = (
            "ffmpeg -y"
            + input_files
            + filter_complex
            + " -map '[v]' -c:v libx264 -preset medium -crf 40 -threads:v 7 "
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
            print("\n\n" + dt.datetime.now().strftime("%y-%m-%d__%H-%M-%S") + " -------- ERROR: \n")
            print("Savepath: " + new_filename)
            print("Command: " + command)
            print("Output of subprocess: \n")
            print(e.output)
            quit()

        current_start_file += len(interest)
