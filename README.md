# Introduction 
This is the codebase for the fish sonar project. The goal of this project is to provide a continuous 
sonar based fish detection solution for HTU. To that end a sonar sensor is placed in the water and connected to a 
raspberry pi via a HDMI capture device. In the future, the raspberry pi shall receive the stream of sonar images, run 
the fish detection algorithm on the data and store the output as well as record the stream of sonar images. To 
facilitate continuous operation the raspberry pi shall send regular heartbeats to a hardware- and possibly a 
cloud-based watchdog. If an issue occurs specific users should be informed. 

# Stucture
The codebase is structured into the following sections:
- **algorithm**: the fish detection algorithm, taking a video file as an input and giving .csv / visual / video 
  output. This part of the code shall be open-sourced at a later stage of the project. This code is ready for a review.
- **analysis**: all code related to applications of the algorithm for development, production or special 
  projects. This section stores the input, output files and tools - it does not need reviews at this stage.
- **hardware**: all code pertaining to the initialization and continous operation of the raspberry pi. This code is 
  still at a prototyping stage and must not be reviewed yet.

# Running the algorithm
This is a short guide to run the fish detection algorithm on a sample video.
1.	Install the requirements specified in requirements_mac.txt
2.	Download the sample sonar video from here: [demo_sample_sonar_recording.mp4 - Azure link valid until 31.03](https://axh4lab4appl4sonar4sa.blob.core.windows.net/sonar-recording-sample/demo_sample_sonar_recording.mp4?sp=r&st=2023-02-02T12:37:44Z&se=2023-03-31T19:37:44Z&spr=https&sv=2021-06-08&sr=b&sig=gw5GanJeONyhg9bcVtagfeAXa2tn7YDHj67GjvlAA8U%3D) 
3.	Place the video in the following folder: _"analysis/demo/"_
4. Specify the desired settings for the algorithm and the output in _"analysis/demo/demo_settings.yaml"_
5. Run the algorithm using _"algorithm/run_algorithm.py"_. E.g.: _"python3 algorithm/run_algorithm.py -yf 
   analysis/demo/demo_settings.yaml"_

# Comments
Tested with Python 3.10, 
direct feedback to Leiv Andresen, leiv.andresen@axpo.com 