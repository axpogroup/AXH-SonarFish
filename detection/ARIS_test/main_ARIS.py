import cv2 as cv
import numpy as np
from FishDetector_ARIS import FishDetector

if __name__ == "__main__":
    enhanced = False
    # recording_file = (
    #         "ARIS_test/2022-06-21_050000_1406_2480 zu besprechen abstieg oder fressen.mp4"
    # ) # interessant 2
    recording_file = (
        "ARIS_test/recordings/2022-06-23_134500_1571_1942 2 Forellen 20-25cm Rechenkontakte "
        "zu besprechen.mp4"
    )

    # recording_file = "output/normed_120_10_std_dev_threshold_2_median_11_drop_duplicates_crop.mp4"  # enhanced
    # recording_file = "output/components/final_old_moving_average_5s.mp4"  # enhanced
    # recording_file = (
    #     "recordings/new_settings/22-11-14_start_17-06-59_crop_swarms_single.mp4"
    # )

    write_file = True
    # output_file = "output/components/normed_120_10_std_dev_threshold_2_median_11_schwarm_temp.mp4"
    output_file = (
        "output/productialization/ARIS/"
        + (recording_file.split("/")[-1]).split(".mp4")[0]
        + "_detection_100_assoc_80.mp4"
    )
    # output_file = "output/normed_120_minus_10.mp4"
    # output_file = "output/normed_120_10_std_dev_threshold_2.mp4"

    # Initialize Input
    video_cap = cv.VideoCapture(recording_file)
    frame_by_frame = False
    previous_img = False

    # grab the width, height, fps and length of the video stream.
    frame_width = int(video_cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv.CAP_PROP_FPS))
    frames_total = int(video_cap.get(cv.CAP_PROP_FRAME_COUNT))

    # initialize the FourCC and a video writer object
    fourcc = cv.VideoWriter_fourcc("m", "p", "4", "v")
    video_writer = cv.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    # Initialize FishDetector Instance
    detector = FishDetector(recording_file)

    frame_no = 0
    while video_cap.isOpened():
        ret, raw_frame = video_cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Start timer
        frame_no += 1
        timer = cv.getTickCount()

        # Detection
        downsample = True
        if enhanced:
            enhanced_frame = raw_frame[:1080, :, :]
            detector.process_frame(raw_frame[1080:, :, :], secondary=enhanced_frame)
        else:
            detector.process_frame(raw_frame, downsample=downsample)

        # Output
        four_images = True
        fullres = False
        if enhanced:
            disp = np.concatenate(
                (
                    detector.draw_output(detector.current_output, debug=True),
                    detector.draw_output(detector.current_raw, classifications=True),
                )
            )
        elif four_images:
            try:
                up = np.concatenate(
                    (
                        detector.retrieve_frame(detector.current_enhanced),
                        detector.retrieve_frame(detector.current_blurred_enhanced),
                    ),
                    axis=1,
                )
                down = np.concatenate(
                    (
                        detector.draw_output(
                            detector.retrieve_frame(detector.current_raw), debug=False
                        ),
                        detector.retrieve_frame(detector.current_threshold),
                    ),
                    axis=1,
                )
                disp = np.concatenate((up, down))
                disp = detector.draw_output(
                    detector.resize_img(disp, 300), only_runtime=True, runtiming=True
                )
            except ValueError:
                disp = raw_frame

        elif fullres:
            disp = detector.draw_output(
                raw_frame, classifications=True, runtiming=True, fullres=True
            )

        else:
            disp = np.concatenate(
                (
                    detector.draw_output(
                        detector.current_enhanced, debug=True, runtiming=True
                    ),
                    detector.draw_output(
                        detector.current_raw, classifications=True, runtiming=True
                    ),
                )
            )
        cv.imshow("frame", disp)

        if write_file:
            video_writer.write(disp)
        if frame_no % 20 == 0:
            print(f"Processed {frame_no/frames_total*100} % of video.")
            if frame_no / frames_total * 100 > 35:
                pass

        if not frame_by_frame:
            usr_input = cv.waitKey(1)
        if usr_input == ord(" "):
            if cv.waitKey(0) == ord(" "):
                frame_by_frame = True
            else:
                frame_by_frame = False
            print("Press any key to continue ... ")
        if usr_input == 27:
            break

    video_cap.release()
    video_writer.release()
    cv.destroyAllWindows()
