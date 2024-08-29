import datetime as dt
import logging
import os
from logging.handlers import TimedRotatingFileHandler

import pymsteams
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

from algorithm.FishDetector import FishDetector
from algorithm.InputOutputHandler import InputOutputHandler

TEAMS_LINK = (
    "https://axpogrp.webhook.office.com/webhookb2/1c7d8e30-f530-4faf-acc0-7ef098a2a388@8619c67c-945a-48ae-8e77-"
    "35b1b71c9b98/IncomingWebhook/c065e1582cc24039a640a25ba0b953e7/adfeab72-7d9c-4e19-a21b-6781b139b707"
)
RED = "FF0000"
GREEN = "00FF00"



def get_logger(log_directory, nametag):
    logfolder_name = nametag + "_logs_session_" + dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")
    logfolder_name = logfolder_name.replace(":", "-")
    os.makedirs(os.path.join(log_directory, logfolder_name), exist_ok=True)
    logger = logging.getLogger(f"{nametag}_logger")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    )
    fh = TimedRotatingFileHandler(
        os.path.join(log_directory, logfolder_name, f"{nametag}.log"),
        when="H",
        interval=12,
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


class CloudHandler:
    def __init__(self):
        load_dotenv()
        connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.blob_service_client = BlobServiceClient.from_connection_string(connect_str)

        self.teams_message = pymsteams.connectorcard(TEAMS_LINK)

    def upload_file_to_container(self, filepath, container_name):
        blob_client = self.blob_service_client.get_blob_client(
            container=container_name,
            blob=filepath,
        )

        with open(file=filepath, mode="rb") as data:
            blob_client.upload_blob(data, overwrite=True, connection_timeout=900)

    def send_message(self, color, title, text):
        self.teams_message.title(title)
        self.teams_message.text(text)
        if color == "red":
            self.teams_message.color(RED)
        elif color == "green":
            self.teams_message.color(GREEN)
        self.teams_message.send()
