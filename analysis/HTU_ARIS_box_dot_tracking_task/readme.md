_Code written by Leiv Andresen, HTD-A, leiv.andresen@axpo.com
January 2023_

_This code is not written or documented for reuse out of the box. Certain settings might will need to be adjusted for successful operation._

The goal is to extract the position of fishes and their interactions with the Rake (Rechen) from ARIS sonar imagery.
To this end HTU marked the relevant content with a program: boxes of different colors (red, blue, green) symbolize fish, green and red dots mark interactions with the rake.
The following code is based on the sonar fish detection algorithm and detects the location of the boxes and dots. 

The input videos must satisfy the following requirements:

- Provide a folder with the video names. Exports of the same sonar video need to have the same prefix
    (date, starting frame and ending frame number) ideally a space separating the prefix from the suffix of the filename.
- No objects must intersect. Intersections will be detected as "issues" and the affected frames
    of the video will be exported to an output video. If the output video is empty,
    then there was no intersection problems.
- If dots are visible from the beginning they will be recorded as appearing at the beginning of the frame,
    therefore they should appear in the course of the video in case the time of occurence is important.
- The script will create csvs for each sonar video and put the detections from every separate export inside it,
    therefore nothing should occur twice in different exports of the same video.
- Each continous occurence of a box and each dot is denoted with a unique ID.

**HOW TO RUN THE PIPELINE:**

1. Start by exporting the paths and locations of the boxes and dots with 1_main_box_and_dot_multiple_videos.py
2. Use 2_label_keypoints_for_transformation.py to extract the same two points from all videos and store them to a .csv file. The frames from which the points are extracted can be stored for future reference.
3. Use 3_transform_coordinates.py to transform the coordinates using the keypoints from step 3. Certain parameters of the transforms need to be adjusted depending on the points that are selected (Search for comments: ADJUST in the code)
4. Use 4_validate_and_plot_detections.ipynb to validate the output and make sure e.g. that all contacts with the rake occur on the rake

A zip file with the code and raw data can be found on mingle: https://mingle.axpo.com/x/aQQNDQ 

_Tested with the following versions: 

python 3.10
opencv-python==4.6.0.66
matplotlib==3.6.2
pyyaml==6.0
pandas=1.5.0

Runtime of 1_main_box_and_dot_multiple_videos.py on 14" MacBook Pro: 30 minutes._