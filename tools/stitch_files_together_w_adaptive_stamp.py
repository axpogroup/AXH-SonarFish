import datetime as dt
import glob
import os
import subprocess

from dateutil.relativedelta import relativedelta


#  ---- CHECKING TIME DIFFS BETWEEN VIDEOS ----
def check_time_gaps(filename_list, date_format):
    time_diffs = []
    for i in range(1, len(filename_list)):
        datetime_start_2 = dt.datetime.strptime(
            os.path.split(filename_list[i])[-1], date_format
        )
        datetime_end_1 = dt.datetime.strptime(
            os.path.split(filename_list[i - 1])[-1], date_format
        ) + relativedelta(minutes=10)
        delta_s = (datetime_start_2 - datetime_end_1).total_seconds()
        time_diffs.append(delta_s)

    time_diffs.sort()
    print("Biggest time gaps: ", time_diffs[-10:])


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
    save_dir = (
        "/Users/leivandresen/Documents/PROJECTS/SONAR_FISH/Field_test_Stroppel_20to24_10_22/"
        "weekend_backup/stitched_final/"
    )
    os.makedirs(name=save_dir, exist_ok=True)
    n_videos_to_join = 24

    filenames = glob.glob(input_files_local)
    filenames.sort()

    current_start_file = 3 + 24 + 24
    epoch = dt.datetime.utcfromtimestamp(0)
    while current_start_file < len(filenames):
        interest = filenames[current_start_file:]
        if len(filenames[current_start_file:]) > n_videos_to_join:
            interest = filenames[
                current_start_file : (current_start_file + n_videos_to_join)
            ]

        date_fmt = "%y-%m-%d_start_%H-%M-%S.mp4"
        datetime_start = dt.datetime.strptime(os.path.split(interest[0])[-1], date_fmt)
        datetime_end = dt.datetime.strptime(
            os.path.split(interest[-1])[-1], date_fmt
        ) + relativedelta(minutes=10)
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
            " -filter_complex "
            + '"'
            + filter_complex
            + final_streams
            + f"concat=n={len(interest)}:v=1:a=0[v]"
            + '"'
        )

        command = (
            "ffmpeg -y"
            + input_files
            + filter_complex
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
            quit()

        current_start_file += len(interest)
