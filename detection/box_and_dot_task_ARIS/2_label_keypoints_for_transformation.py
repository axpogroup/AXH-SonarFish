# Code written by Leiv Andresen, HTD-A, leiv.andresen@axpo.com

import csv
import glob

import cv2 as cv

global raw_frame
global csv_line


def printCoordinate(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(raw_frame, (x, y), 20, (0, 0, 255), 3)
        strXY = "(" + str(x) + "," + str(y) + ")"
        font = cv.FONT_HERSHEY_PLAIN
        cv.putText(raw_frame, strXY, (x + 40, y - 40), font, 1, (100, 100, 255))
        cv.imshow("frame", raw_frame)
        csv_line.append(x)
        csv_line.append(y)


if __name__ == "__main__":
    # Specify the input folder, possibly ADJUST
    filenames = glob.glob("ARIS_videos/2022/" + "*.mp4")
    filenames.sort()
    print("Found the following files: \n")
    for file in filenames:
        print(file)
    print("\n")

    print("Creating new csv...")

    # Specify the output file, ADJUST
    csv_file = "output/keypoints_2022_DESCRIPTION.csv"
    csv_f = open(csv_file, "w")
    csv_writer = csv.writer(csv_f)
    header = [
        "filename",
        "point 1x",
        "point 1y",
        "point 2x",
        "point 2y",
        "water level x",
        "water level y",
    ]

    validation_images = []

    csv_writer.writerow(header)
    finish_frame = 5
    for file in filenames:
        print(f"\nProcessing  {file}")
        # global csv_line = []
        csv_line = []
        csv_line.append(file)

        video_cap = cv.VideoCapture(file)

        frame_no = 0
        while video_cap.isOpened():
            ret, raw_frame = video_cap.read()
            # if frame is read correctly ret is True
            raw_frame = cv.resize(raw_frame, (1920, 1080))
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            if frame_no == finish_frame:
                cv.putText(
                    raw_frame,
                    "Click two points in a line, press any key for next video ...",
                    (50, 50),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (200, 200, 255),
                    2,
                )
            if frame_no == finish_frame + 1:
                break

            # Detection
            cv.imshow("frame", raw_frame)
            if frame_no >= finish_frame:
                validation_images.append(raw_frame)
                cv.setMouseCallback("frame", printCoordinate)
                usr_input = cv.waitKey(0)
            else:
                usr_input = cv.waitKey(1)

            frame_no += 1

        video_cap.release()
        cv.destroyAllWindows()

        csv_writer.writerow(csv_line)

    csv_f.close()
