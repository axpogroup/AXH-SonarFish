import datetime
import io
import os
import random
import string

import pandas as pd
from azure.storage.blob import (
    BlobClient,
    BlobSasPermissions,
    BlobServiceClient,
    ContentSettings,
    generate_blob_sas,
)

# TODO
# Luca make blob storage account and container / terraform?
# Env variables: (put in .bashrc file on rpi), on mac it is .zshrc
# upload some files


class BlobStorageHandler:
    def __init__(self, container_name):
        self.storage_account = os.environ["azureBlobAccountName"]
        self.connection_string = os.environ["azureBlobConnectionString"]
        self.access_key = os.environ["azureBlobAccessKey"]
        self.container_name = container_name
        self.blob_name = None

    def create_blob_name(self, ts_start, ts_end, resampling_method, file_format):
        ts_start = ts_start.isoformat()
        ts_end = ts_end.isoformat()
        random_str = "".join(
            random.choices(string.ascii_uppercase + string.digits, k=5)
        )
        if file_format is None:
            file_format = ".xlsx"
        blob_name = (
            "export_"
            + ts_start
            + "_"
            + ts_end
            + "_"
            + resampling_method
            + "_"
            + random_str
            + file_format
        )
        return blob_name

    def download_file_into_df(self, conn_str, blob_name, file_type):
        blob_client = BlobClient.from_connection_string(
            conn_str=conn_str, container_name=self.container_name, blob_name=blob_name
        )
        if file_type == ".parquet":
            df = pd.read_parquet(
                io.BytesIO(blob_client.download_blob().content_as_bytes())
            )
        elif file_type == ".xlsx":
            df = pd.read_excel(
                io.BytesIO(blob_client.download_blob().content_as_bytes())
            )
        elif file_type == ".csv":
            df = pd.read_csv(io.BytesIO(blob_client.download_blob().content_as_bytes()))
        else:
            raise ValueError(
                f"Can only download parquet or xlsx files. {file_type} was requested."
            )
        return df

    def upload_file(
        self,
        df,
        ts_start=None,
        ts_end=None,
        resampling_method=None,
        file_format=".xlsx",
    ):
        blob_service_client = BlobServiceClient.from_connection_string(
            conn_str=self.connection_string
        )
        if self.blob_name is None:
            self.blob_name = self.create_blob_name(
                ts_start, ts_end, resampling_method, file_format
            )

        blob_client = blob_service_client.get_blob_client(
            container=self.container_name, blob=self.blob_name
        )

        writer = io.BytesIO()
        if file_format == ".csv":
            df.to_csv(writer, na_rep="#N/A")
            content_setting = ContentSettings(content_type="application/CSV")
        else:
            df.to_excel(writer, na_rep="#N/A")
            content_setting = ContentSettings(
                content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        blob_client.upload_blob(
            writer.getvalue(), overwrite=True, content_settings=content_setting
        )
        writer.close()

    def get_download_url(self):
        blob_sas_token = generate_blob_sas(
            account_name=self.storage_account,
            container_name=self.container_name,
            blob_name=self.blob_name,
            account_key=self.access_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.datetime.utcnow() + datetime.timedelta(days=30),
        )
        download_url = (
            "https://"
            + self.storage_account
            + ".blob.core.windows.net/"
            + self.container_name
            + "/"
            + self.blob_name
            + "?"
            + blob_sas_token
        )
        return download_url


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    container_test = "test"
    blob_storage = BlobStorageHandler(container_name=container_test)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
