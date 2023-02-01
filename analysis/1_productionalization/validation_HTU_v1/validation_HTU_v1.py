import datetime as dt
import glob
import subprocess

if __name__ == "__main__":
    video_files = "analysis/1_productionalization/validation_HTU_v1/recordings/*.mp4"

    files = glob.glob(video_files)
    files.sort()

    print(files)

    for file in files:
        if "22-11-16_start_02-23-35" not in file:
            continue
        command = (
            f"python3 algorithm/run_algorithm.py -yf 'analysis/1_productionalization/validation_HTU_v1/"
            f"settings/validation_HTU_v1_settings_ARIS.yaml' -if {file}"
        )

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
                + file
                + "\n\n"
            )
        except subprocess.CalledProcessError as e:
            print(
                "\n\n"
                + dt.datetime.now().strftime("%y-%m-%d__%H-%M-%S")
                + " -------- ERROR: \n"
            )
            print("File: " + file)
            print("Command: " + command)
            print("Output of subprocess: \n")
            print(e.output)
            quit()
