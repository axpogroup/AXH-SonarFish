# Introduction 
The fish sonar tracks fish and floating debris in front of run-of-river powerplants and classifies the trajectories into either fish or object. Based on the tracked fish, a bypass in the run-of-river powerplant can be opened to allow the fish to travel downstream safely. The current version of the fishsonar is not set up for real time classification of trajectories but instead runs tracking, saves trajectories, and then classifies the trajectories in batch.

# Sample Results - Powerplant Stroppel
<p align="center">
  <img src="data/sample_tracking/stroppel.gif" alt="Fish Detection Demo" width="900">
</p>

# Stucture
The codebase is structured into the following sections:
- **algorithm**: The fish detection algorithm, taking a video file as an input and giving .csv / visual / video 
  output for trajectories. 
- **analysis**: All code associated with the classification of trajectories into the categories fish and 
- **continous_operation (deprecated)**: Code pertaining to the setup, initialization and continuous operation of the 
  Raspberry Pi for tracking in Stroppel. This part is deprecated since for Lavey and all future sonar installations, a more professional setup for video capture and storage will be used.

## Data Structure
The data structure is as follows:

    - data
        - labels
        - model_output
        - raw
          - videos
          - labels 

- *model_output* : Contains the output of the fish detection algorithm as video and csv. 
- *raw*: contains the raw video files.
  - videos: Contains the raw video files.
  - labels: Contains the labels video files. If a csv with labeled fish tracks exists in this directory, they are read and displayed in the output video. This way videos with labeled trajectories can be generated

# Environment Setup
- Install Requirements from requirements.txt
- Add .env file with the following contents in the top level folder:
    ```
        RESOURCE_GROUP=<your-azure-resource-group>
        WORKSPACE_NAME=<your-azure-ml-workspace>
        SUBSCRIPTION_ID=<your-azure-subscription-id>
  ```
- Create a virtual environment:
  ```bash
    uv venv --python 3.11 .venv
  ```
- Install dependencies
  ```bash
    source .venv/bin/activate
    uv pip install -r requirements.dev.txt
  ```
- Add the path to the demo yaml settings to your launch.json in VS-Code
    "env": {
        "PYTHONPATH": "${workspaceFolder}:${workspaceFolder}/analysis:${env:PYTHONPATH}",
    },
    "args": [
        "--yaml_file", "settings/settings_stroppel.yaml",
    ]
  The processing oftentimes takes some seconds to start. Once the video pops up, it will take several seconds more of processing video material for tracks to show up. These tracks do not contain labels since tracking and classification are split into two different steps.
- Install `ffmpeg` for compression of the mp4 (this might take several minutes)
  ```bash
  sudo apt install ffmpeg
  ```
  or with brew
  ```bash
    brew install ffmpeg
  ```

# Running the algorithm

## Run on a sample video
You can now run the `algorithm/main_algorithm.py` file in debugging mode in VS-code or with 
  ```bash
  python algorithm/run_algorithm.py --yaml_file settings/settings_stroppel.yaml
  ```

## Run on a batch of files and evaluating the output on Azure-ML
You can run the tracking and classification algorithms as Azure-ML pipelines with the [run_on_azure/](run_on_azure/launch_kalman_tracking_azure.ipynb) notebook. The notebook has different pipelines that either launch only tracking, tracking and classification, or tracking, classification, and labeling of videos with labeled trajectories. For classification you will need ground truth data for training the classification model and reference the correct path on your Azure ML workspace.

# Running Tests
- For now, tests have to be executed from the tests folder.
  - This is due to the fact that the algorithm relies on relative paths, and this behaviour should be tested in the tests
- use this to run it: ```export PYTHONPATH=${PYTHONPATH}:$(pwd); cd tests; pytest```

# How to Contribute

# Creating Ground Truth Data
The easiest way to create ground truth data is to run the tracking on the videos you want to use to extract ground truth trajectories. You can run the tracking and select the dual output mode in the tracking settings yaml. This will save a single video with two versions of the sonar video next to eachother, the original and the video with tracks and ids displayed. Noting down the tracks of fish in a csv can then be joined onto the output track csv to have tracks with labels.

An alternative approach is the use of a video editing tool that allows to track a trajectory with your mouse pointer. We have found this approach more cumbersome than the one described above.

# Acknowledgements
Development of this project was financed by the Swiss Bundesamt für Umwelt BAFU. Further details can be found in the [notice file](NOTICE).