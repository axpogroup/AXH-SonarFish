import cv2 as cv


class VideoHandler:
    def __init__(self, settings_dict):
        self.settings_dict = settings_dict
        self.video_cap = cv.VideoCapture(self.settings_dict["input_file"])

        if "record_output_video" in settings_dict.keys():
            self.initialize_output_recording()

        self.frame_by_frame = False
        self.usr_input = None
        self.frame_no = 0
        self.frames_total = int(self.video_cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.video_cap.get(cv.CAP_PROP_FPS))

        self.current_raw_frame = None

    def initialize_output_recording(self):
        # grab the width, height, fps and length of the video stream.
        frame_width = int(self.video_cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.video_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.video_cap.get(cv.CAP_PROP_FPS))

        # initialize the FourCC and a video writer object
        fourcc = cv.VideoWriter_fourcc("m", "p", "4", "v")
        self.video_writer = cv.VideoWriter(
            self.settings_dict["record_output_video"],
            fourcc,
            fps,
            (frame_width, frame_height),
        )
        return

    def get_new_frame(self):
        if self.video_cap.isOpened():
            ret, self.current_raw_frame = self.video_cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                self.video_cap.release()
                if "record_output_video" in self.settings_dict.keys():
                    self.video_writer.release()
                cv.destroyAllWindows()
                return False

            return True

        else:
            print("ERROR: Video Capturer is not open.")
            return False

    def show_image(self, img, playback_controls=False):
        cv.imshow("frame", img)

        if self.frame_no % 20 == 0:
            print(f"Processed {self.frame_no / self.frames_total * 100} % of video.")
        self.frame_no += 1

        if not self.frame_by_frame:
            self.usr_input = cv.waitKey(1)

        # If pause button was pressed, then the pause button can be used to playback frame by frame.
        # Any other button will resume playback
        if self.usr_input == ord(" "):
            print("Press any key to continue playback or SPACE for next frame ... ")
            if cv.waitKey(0) == ord(" "):
                self.frame_by_frame = True
            else:
                self.frame_by_frame = False
            return

        if self.usr_input == 27:
            print("User pressed ESC, closing video stream ...")
            self.video_cap.release()
            if "record_output_video" in self.settings_dict.keys():
                self.video_writer.release()
            cv.destroyAllWindows()
            return

    def write_image(self, img):
        if "record_output_video" in self.settings_dict.keys():
            self.video_writer.write(img)
