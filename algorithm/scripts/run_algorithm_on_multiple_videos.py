import argparse
import os
from pathlib import Path

import yaml
from azureml.core import Workspace

from algorithm.run_algorithm import main_algorithm

if __name__ == "__main__":
    argParser = argparse.ArgumentParser(description="Run the fish detection algorithm with a settings .yaml file.")
    argParser.add_argument("-ct", "--config_track", help="path to the YAML settings file", required=True)
    argParser.add_argument("-cj", "--config_job", help="path to the YAML with AzureML job settings", required=True)
    args = argParser.parse_args()

    with open(args.yaml_file) as f:
        settings = yaml.load(f, Loader=yaml.SafeLoader)
        if args.input_file is not None:
            print("replacing input file.")
            settings["file_name"] = args.input_file

    workspace = Workspace(
        resource_group=os.getenv("RESOURCE_GROUP"),
        workspace_name=os.getenv("WORKSPACE_NAME"),
        subscription_id=os.getenv("SUBSCRIPTION_ID"),
    )
    input_directory_path = (
        Path(
            "azureml://subscriptions/your-azure-subscription-id/resourcegroups/"
            "axsa-lab-appl-fishsonar-rg/workspaces/axsa-lab-appl-fishsonar-ml/datastores/"
            "workspaceblobstore/paths/stroppel_videos/"
        ),
    )

    input_video_file_paths = input_directory_path.glob("**/*.mp4")
    for input_video_path in input_video_file_paths:
        settings["file_name"] = input_video_path.name
        main_algorithm(settings)
