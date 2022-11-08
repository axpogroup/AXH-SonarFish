import glob
import os
import subprocess
import time

from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

recording_files_mac = (
    "/Users/leivandresen/Documents/PROJECTS/SONAR_FISH/Field_test_Stroppel_20to24_10_22/"
    "weekend_backup/"
)
recording_files_rpi = "/media/fish-pi/sonar-disk1/Stroppel_ongoing/"
folder_for_snippets = "./recording_snippets"
os.makedirs(name=folder_for_snippets, exist_ok=True)

recordings = glob.glob(recording_files_mac + "*.mp4")
recordings.sort()
snippet_name = os.path.splitext(os.path.split(recordings[-2])[-1])[0] + "_snippet.mp4"
snippet_path = os.path.join(folder_for_snippets, snippet_name)
snippet_cmd = f"ffmpeg -y -i {recordings[-2]} -c:v libx264 -t 00:00:10 {snippet_path}"
success = False
try:
    output = subprocess.run(
        snippet_cmd,
        check=True,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(output.stdout)
    print("Snippet created: ", snippet_path)
    success = True
except subprocess.CalledProcessError as e:
    print("-------- ERROR making snippet. ---------")
    print("Original file: " + recordings[-1])
    print("Command: " + snippet_cmd)
    print("Output of subprocess: \n")
    print(e.output)

time.sleep(2)

# Upload the snippet
if success:
    try:
        print("Preparing to upload snippet...")
        load_dotenv()
        connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

        # Create the BlobServiceClient object
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)

        # Create a blob client using the local file name as the name for the blob
        container_name = "test"
        blob_client = blob_service_client.get_blob_client(
            container=container_name, blob=snippet_name
        )

        print("\nUploading to Azure Storage as blob:\n\t" + snippet_path)

        # Upload the created snippet
        with open(file=snippet_path, mode="rb") as data:
            blob_client.upload_blob(data)

    except Exception as ex:
        print("Exception:")
        print(ex)
