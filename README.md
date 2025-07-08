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

# Running Locally
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
- install dependencies
  ```bash
    source .venv/bin/activate
    uv pip install -r requirements.dev.txt
  ```
- add the path to the demo yaml settings to your launch.json in VS-Code
    "env": {
        "PYTHONPATH": "${workspaceFolder}:${workspaceFolder}/analysis:${env:PYTHONPATH}",
    },
    "args": [
        "--yaml_file", "settings/settings_stroppel.yaml",
    ]
  The processing oftentimes takes some seconds to start. Once the video pops up, it will take several seconds more of processing video material for tracks to show up. These tracks do not contain labels since tracking and classification are split into two different steps.
- install `ffmpeg` for compression of the mp4 (this might take several minutes)
  ```bash
  sudo apt install ffmpeg
  ```
  or with brew
  ```bash
    brew install ffmpeg
  ```

# Running Tests
- For now, tests have to be executed from the tests folder.
  - This is due to the fact that the algorithm relies on relative paths, and this behaviour should be tested in the tests
- use this to run it: ```export PYTHONPATH=${PYTHONPATH}:$(pwd); cd tests; pytest```

# Running the algorithm

## Run on a batch of files and evaluating the output

1. Store your prelabeled videos in the folder `data/raw/labels`
2. If you want to reduce the frame rate of the videos, run the script `analysis/reduce_frame_rate.py`. This script needs `settings/preprocessing_settings.yaml` file where you have to choose:
   1. fps: the desired frame rate
   You will need to do this both for the raw videos and the labeled videos in order for the labels to match.
3. To extract the labels in csv run `algorithm/scripts/extract_labels_from_videos.py` with settings file tracking_box_settings.yaml.
   in the folder `data/intermediate/labels`
4. Run the algorithm for one video using `algorithm/run_algorithm.py`. E.g.: `python3 algorithm/run_algorithm.py -yf 
   settings/demo_settings.yaml`. Specify the name of the video file in the settings file. 
5. Use the settings "display_output_video: True" and "display_mode_extensive: True" to tune the settings in an 
   interactive window and "display_mode_extensive: False" to read the velocity of the river. 
6. Evaluate the output using the script `algorithm/validation/mot16_metrics.py`.


## Run on a sample video
1. Download the sample sonar video from here: [demo_sample_sonar_recording.mp4 - Sharepoint](https://axpogrp.sharepoint.com/:v:/s/DEPTHTD-A/ESdKpDEWDEBDqYR6KVFZ0D8BJrxKcDi6F8JaenjD0YhWWw?e=5jNCLF) 
2. Place the video in the following folder: `data/raw/videos/`
3. Do the same steps as in the previous section and comment out the ground_truth_directory in the settings file. This will cause the algorithm to not compare the output with the ground truth.

# Labeling Videos
We use the video editing tool shotcut to label videos.
## Converting Colormaps
You can convert the red colormap the sonar videos are stored in to a jet colormap, which is easier to interpret for the human eye. The script, `convert_video_red_to_jet.py`, does this transformation.
You can (after activating the `.venv` of the project) run the script with
```bash
python .algorithm/convert_video_red_to_jet.py </path/to/your/video.mp4>
```
Replace `</path/to/your/video.mp4>` with the path to the video file you want to process.
You can also specify the boosting $\alpha$ and $\beta$ for the conversion with the `--boosting_alpha` and `--boosting_beta` flags. The defaults are 2.0 and 30.0 respectively.

# Comments
Tested with Python 3.10, 
direct questions to the Axpo HTD-A team 