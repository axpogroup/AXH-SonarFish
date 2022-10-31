import glob

import numpy as np

if __name__ == "__main__":
    # This script changes the filenames since the raspberry pi had the wrong system time
    files_stick = "/Volumes/SONAR_STICK/recordings_stroppel_weekend/*.mp4"
    files_local = (
        "/Users/leivandresen/Documents/PROJECTS/SONAR_FISH/Field_test_Stroppel_20to24_10_22/"
        "weekend_backup/*.mp4"
    )
    files_test = "/Users/leivandresen/Documents/Hydro_code/AXH-SonarFish/old_recording_handler_tests/*.mp4"

    filenames = glob.glob(files_local)
    filenames.sort()
    interest = filenames[3:27]
    for file in interest:
        print(file)

    print("-----")

    "[0:0][1:0][2:0][3:0][4:0][5:0]concat=n=6:v=1:a=0[v]"
    input_files = "".join([(" -i " + file) for file in interest])
    input_params = (
        "'"
        + "".join([f"[{file_no}:0]" for file_no in np.arange(0, len(interest))])
        + f"concat=n={len(interest)}:v=1:a=0[v]"
        + "'"
    )
    command = (
        "ffmpeg "
        + input_files
        + " -filter_complex "
        + input_params
        + " -map '[v]' -c:v libx264 -preset medium -crf 40 22-10-20_start_17_51_end_21_51_compressed.mp4"
    )

    print(command)
