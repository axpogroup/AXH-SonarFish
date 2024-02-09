# Introduction 
This is the codebase for the fish sonar project. The goal of this project is to provide a continuous 
sonar based fish detection solution for HTU. To that end a sonar sensor is placed in the water and connected to a 
Raspberry Pi via a HDMI capture device. The Raspberry Pi receives the stream of sonar images, runs 
the fish detection algorithm on the data and stores the output and records the stream of sonar images. To 
facilitate continuous operation the Raspberry Pi sends regular heartbeats to a hardware watchdog. At a later stage a 
Grafana dashboard could act as cloud-based watchdog and alert users if the system is not running as expected.

# Stucture
The codebase is structured into the following sections:
- **algorithm**: the fish detection algorithm, taking a video file as an input and giving .csv / visual / video 
  output. This part of the code shall be open-sourced at a later stage of the project and can function independently 
  of the rest of the continuous operation setup.
- **analysis**: all code related to applications of the algorithm for development or special 
  projects. This section stores input files, output files and tools.
- **continous_operation**: all code pertaining to the setup, initialization and continuous operation of the 
  Raspberry Pi.

## Data Structure
The data structure is as follows:

    - data
        - labels
        - model_output
        - intermediate
          - videos
          - labels
        - raw
          - videos
          - labels 



- *labeled_videos* : contains the labeled videos.
- *labels*: this is where the output of *extract_labels_from_videos.py* is stored.
- *model_output* : contains the output of the fish detection algorithm. This includes a folder for each video file 
    containing the .csv file with the fish detections and the visual output.
- *raw*: contains the raw video files.
  - videos: contains the raw video files.
  - labels: contains the labels video files
- *intermediate*: cotaions the output of the *reduce_frame_rate.py* script. This includes a folder for each video file 
    containing the reduced frame rate video file.

# Continuous operation
This is a high-level overview of the steps needed to run the continous operation.
1. Assemble the hardware as described in [Mingle](https://mingle.axpo.com/display/HTD/System+Overview) 
2. Follow the instructions in continous_operation/raspberry_pi_setup_instructions.md to prepare the Raspberry Pi 
   software. This includes installing Ubuntu, git, getting the repo, installing the requirements and setting up 
   autostart on reboot and the watchdog.
3. Specify the desired output directory, detector setup and other settings in continous_operation/settings. When 
   establishing a new location or sonar settings be sure to adapt the settings and masks of the algorithm on a 
   sample recording file using the algorithms interactive visual output.
4. Reboot the system and the recording should start after 2 minutes. 
5. Monitor the recording and the outputs using the commands in continous_operation/src/controller.py. They should be 
   accessible via an alias, e.g., "control check_status". 

# Running the algorithm on an individual file
This is a short guide to run the fish detection algorithm on a sample video.
1.	Install the requirements specified in requirements.txt
2.	Download the sample sonar video from here: [demo_sample_sonar_recording.mp4 - Sharepoint](https://axpogrp.sharepoint.com/:v:/s/DEPTHTD-A/ESdKpDEWDEBDqYR6KVFZ0D8BJrxKcDi6F8JaenjD0YhWWw?e=5jNCLF) 
3.	Place the video in the following folder: _"data/raw/videos/"_
4. Specify the desired settings for the algorithm and the output in _"analysis/demo/demo_settings.yaml"_
5. Run the algorithm using _"algorithm/run_algorithm.py"_. E.g.: _"python3 algorithm/run_algorithm.py -yf 
   analysis/demo/demo_settings.yaml"_
6. Use the settings "display_output_video: True" and "display_mode_extensive: True" to tune the settings in an 
   interactive window and "display_mode_extensive: False" to read the velocity of the river.

# Running the algorithm on a batch of files and evaluating the output

1. Store your prelabeled videos in the folder _"data/raw/labels"_
2. If you want to reduce the frame rate of the videos, run the script _"analysis/reduce_frame_rate.py"_. This script needs _"settings/preprocessing_settings.yaml"_ file where you have to choose:
   1. fps: the desired frame rate
3. To extract the labels in csv run _"algorithm/extract_labels_from_videos.py"_ with settings file tracking_box_settings.yaml.
   in the folder _"data/intermediate//labels"_
4. Run the algorithm for one video using _"algorithm/run_algorithm.py"_. E.g.: _"python3 algorithm/run_algorithm.py -yf 
   analysis/demo/demo_settings.yaml"_. Specify the name of the video file in the settings file. 
5. Evaluate the output using the script _"algorithm/validation/mot16_metrics.py"_.

# Comments
Tested with Python 3.10, 
direct questions to the Axpo HTD-A team 