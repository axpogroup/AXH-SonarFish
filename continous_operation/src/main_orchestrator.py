import datetime as dt
import glob
import os
import sys
import traceback

sys.path.append("/home/fish-pi/code/")
import time

import pandas as pd
import yaml

from continous_operation.src import utils


def modified_in_past_x_minutes(filepath, x):
    if (
        dt.datetime.now(dt.timezone.utc)
        - dt.datetime.fromtimestamp(os.path.getmtime(filepath), tz=dt.timezone.utc)
    ) < dt.timedelta(minutes=x):
        return True
    else:
        return False


def check_recordings():
    all_recordings = glob.glob(
        os.path.join(
            orchestrator_settings_dict["output_directory"], "recordings", "**/*.mp4"
        )
    )
    if len(all_recordings) == 0:
        raise Exception("No recordings found.")

    all_recordings = sorted(all_recordings, key=os.path.getmtime)
    if not modified_in_past_x_minutes(all_recordings[-1], no_mod_thres):
        raise Exception(f"No file modification in the past {no_mod_thres} minutes.")

    try:
        existing_completed_recordings = pd.read_csv(
            os.path.join(
                orchestrator_settings_dict["output_directory"],
                "file_lists",
                "completed_recordings_list.csv",
            )
        )["path"].to_list()
    except FileNotFoundError:
        existing_completed_recordings = []

    new_completed_recordings = [
        rec
        for rec in all_recordings
        if (rec not in existing_completed_recordings)
        and not modified_in_past_x_minutes(rec, no_mod_thres)
    ]

    existing_completed_recordings = (
        existing_completed_recordings + new_completed_recordings
    )
    pd.DataFrame(existing_completed_recordings, columns=["path"]).to_csv(
        os.path.join(
            orchestrator_settings_dict["output_directory"],
            "file_lists",
            "completed_recordings_list.csv",
        ),
        index=False,
    )
    return new_completed_recordings


def detect_on_new_files():
    try:
        processed_recordings = pd.read_csv(
            os.path.join(
                orchestrator_settings_dict["output_directory"],
                "file_lists",
                "processed_recordings_list.csv",
            )
        )["path"].to_list()
    except FileNotFoundError or pd.errors.EmptyDataError:
        processed_recordings = []

    try:
        existing_completed_recordings = pd.read_csv(
            os.path.join(
                orchestrator_settings_dict["output_directory"],
                "file_lists",
                "completed_recordings_list.csv",
            )
        )["path"].to_list()
    except FileNotFoundError or pd.errors.EmptyDataError:
        existing_completed_recordings = []

    try:
        detection_files = pd.read_csv(
            os.path.join(
                orchestrator_settings_dict["output_directory"],
                "file_lists",
                "detection_files_list.csv",
            )
        )["path"].to_list()
    except FileNotFoundError or pd.errors.EmptyDataError:
        detection_files = []

    to_process = [
        rec
        for rec in existing_completed_recordings
        if (rec not in processed_recordings)
    ]

    if len(to_process) == 0:
        return []

    # Only process the latest file, otherwise watchdog might bite
    to_process = sorted(to_process, key=os.path.getmtime)
    recording = to_process[-1]

    logger.info("Detecting fish in recording: " + recording + " ...")
    detections = detection_handler.detect_from_file(recording)

    date_dir = os.path.split(os.path.split(recording)[0])[-1]
    savedir = os.path.join(
        orchestrator_settings_dict["output_directory"], "detections", date_dir
    )
    os.makedirs(name=savedir, exist_ok=True)
    name = os.path.split(recording)[-1][:-4] + ".csv"
    name = os.path.join(savedir, name)

    detections.to_csv(name, index=False)
    detection_files.append(name)
    pd.DataFrame(detection_files, columns=["path"]).to_csv(
        os.path.join(
            orchestrator_settings_dict["output_directory"],
            "file_lists",
            "detection_files_list.csv",
        ),
        index=False,
    )

    logger.info("Success.")
    processed_recordings.append(recording)
    pd.DataFrame(processed_recordings, columns=["path"]).to_csv(
        os.path.join(
            orchestrator_settings_dict["output_directory"],
            "file_lists",
            "processed_recordings_list.csv",
        ),
        index=False,
    )

    return recording


def upload_new_files():
    try:
        uploaded_detections = pd.read_csv(
            os.path.join(
                orchestrator_settings_dict["output_directory"],
                "file_lists",
                "uploaded_detections_list.csv",
            )
        )["path"].to_list()
    except FileNotFoundError or pd.errors.EmptyDataError:
        uploaded_detections = []

    try:
        existing_detections = pd.read_csv(
            os.path.join(
                orchestrator_settings_dict["output_directory"],
                "file_lists",
                "detection_files_list.csv",
            )
        )["path"].to_list()
    except FileNotFoundError or pd.errors.EmptyDataError:
        existing_detections = []

    to_upload = [rec for rec in existing_detections if (rec not in uploaded_detections)]

    # Limit amount of recordings to upload per loop to 10 files
    if len(to_upload) > 10:
        to_upload = to_upload[:10]

    for detection in to_upload:
        cloud_handler.upload_file_to_container(
            detection, orchestrator_settings_dict["azure_container_name"]
        )
        logger.info("Uploaded to Azure Storage as blob: " + detection)
        uploaded_detections.append(detection)
        pd.DataFrame(uploaded_detections, columns=["path"]).to_csv(
            os.path.join(
                orchestrator_settings_dict["output_directory"],
                "file_lists",
                "uploaded_detections_list.csv",
            ),
            index=False,
        )


def upload_logs_of_past_hour():
    log_files = glob.glob(
        os.path.join(
            orchestrator_settings_dict["output_directory"], "logs", "**/*.log*"
        ),
        recursive=True,
    )
    if len(log_files) == 0:
        raise Exception(
            f"No log files found in {os.path.join(orchestrator_settings_dict['output_directory'], 'logs', '**/*.log*')}"
        )

    logs = [log for log in log_files if modified_in_past_x_minutes(log, 60)]
    for log in logs:
        cloud_handler.upload_file_to_container(
            log, orchestrator_settings_dict["azure_container_name"]
        )
        logger.debug("Uploaded to Azure Storage as blob: " + log)


if __name__ == "__main__":
    time.sleep(60)  # Wait for time sync and also for the recording to start

    cwd = "/home/fish-pi/code/continous_operation/"
    # cwd = "/Users/leivandresen/Documents/Hydro_code/AXH-SonarFish/continous_operation"
    with open(os.path.join(cwd, "settings/orchestrator_settings.yaml")) as f:
        orchestrator_settings_dict = yaml.load(f, Loader=yaml.SafeLoader)
    with open(
        os.path.join(cwd, "settings/detector_settings_continous_operation.yaml")
    ) as f:
        detector_settings_dict = yaml.load(f, Loader=yaml.SafeLoader)
    instance_start_dt = dt.datetime.now(dt.timezone.utc)

    logger = utils.get_logger(
        os.path.join(orchestrator_settings_dict["output_directory"], "logs"),
        "orchestrator",
    )
    logger.info(
        f"Starting new instance with the following settings: \n{orchestrator_settings_dict}"
    )
    detection_handler = utils.DetectionHandler(detector_settings_dict)
    os.makedirs(
        os.path.join(orchestrator_settings_dict["output_directory"], "detections"),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(orchestrator_settings_dict["output_directory"], "file_lists"),
        exist_ok=True,
    )

    no_mod_thres = orchestrator_settings_dict[
        "error_after_no_file_modification_minutes"
    ]
    # If the system freezes then there won't be
    # an error teams message, so this indicates, that a new instance started
    initial_teams_message_sent = False

    while True:
        try:
            logger.info("Checking recording status and new files to detect fish.")
            new_files = check_recordings()
            logger.info(f"Recording running. Found {len(new_files)} new files.")
            new_detection = detect_on_new_files()
            logger.info(f"Detected fish on {new_detection} recordings.")

            # Cloud stuff
            if orchestrator_settings_dict["use_cloud"]:
                try:
                    cloud_handler = utils.CloudHandler()
                    upload_new_files()

                    # Send initial heartbeat to MS Teams
                    if not initial_teams_message_sent:
                        cloud_handler.send_message(
                            "green",
                            "Uploaded detections in new session.",
                            f"Instance start time UTC: "
                            f"{instance_start_dt.isoformat(timespec='milliseconds')}",
                        )
                        logger.info("Sent initial Heartbeat to MS Teams.")
                        initial_teams_message_sent = True
                except Exception as file_upload_exception:
                    if orchestrator_settings_dict["raise_exception_on_cloud_error"]:
                        logger.error(
                            "Issue with the cloud."
                        )
                        raise file_upload_exception
                    else:
                        logger.warning(
                            "Issue with the cloud. \n" + "".join(traceback.format_exception(file_upload_exception))
                        )

            # Write to watchdog
            pd.DataFrame(
                [dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")]
            ).to_csv(orchestrator_settings_dict["watchdog_food_file"])
            logger.info("Wrote to watchdog.")
            logger.info(
                f"Sleeping for {orchestrator_settings_dict['sleep_interval_minutes']} minutes."
            )
            time.sleep(int(orchestrator_settings_dict["sleep_interval_minutes"] * 60))

        except Exception as e:
            orchestrating_error = e
            logger.error("An exception occured while orchestrating! \n" + "".join(traceback.format_exception(e)))
            break

    logger.info(
        "Ending execution. Watchdog will not be fed and cause reboot inside 30 min."
    )

    if orchestrator_settings_dict["use_cloud"]:
        try:
            cloud_handler = utils.CloudHandler()
            logger.info("Attempting to send error Message to MS Teams.")
            cloud_handler.send_message(
                "red",
                "ERROR",
                f"UTC Time: "
                f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='milliseconds')}:" +
                "".join(traceback.format_exception(orchestrating_error))
            )
        except Exception as e:
            logger.error("Error sending Message to MS Teams. \n" + "".join(traceback.format_exception(e)))
        try:
            cloud_handler = utils.CloudHandler()
            logger.info("Attempting to upload logs.")
            upload_logs_of_past_hour()
        except Exception as e:
            logger.error("Error sending uploading logs. \n" + "".join(traceback.format_exception(e)))
