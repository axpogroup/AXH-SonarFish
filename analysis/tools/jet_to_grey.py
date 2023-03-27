import cv2 as cv
import numpy as np
from matplotlib._cm import _jet_data
from matplotlib.colors import LinearSegmentedColormap

# This code was copied from a forum and can be used to convert a jet colormap value to grayscale. However it is extremely slow.

def initialize_model():
    cm = LinearSegmentedColormap("jet", _jet_data, N=2**8)
    # cm = colormaps['turbo'] swap with jet if you use turbo colormap instead

    cm._init()  # Must be called first. cm._lut data field created here

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    fm = cv.FlannBasedMatcher(index_params, search_params)

    # JET, BGR order, excluding special palette values (>= 256)
    fm.add(255 * np.float32([cm._lut[:256, (2, 1, 0)]]))  # jet
    fm.train()
    return fm


def jet_to_gray(fm, img):
    (height, width) = img.shape[:2]

    # look up all pixels
    query = img.reshape((-1, 3)).astype(np.float32)
    matches = fm.match(query)

    # statistics: `result` is palette indices ("grayscale image")
    output = np.uint16([m.trainIdx for m in matches]).reshape(height, width)
    result = np.where(output < 256, output, 0).astype(np.uint8)
    # dist = np.uint8([m.distance for m in matches]).reshape(height, width)

    return result  # , dist uncomment if you wish accuracy image


if __name__ == "__main__":
    recording_file = "recordings/22-10-20_start_18_29_snippet.mp4"
    many_settings_file = "recordings/12-03_verschied_Einstellungen.m4v"
    swarm_file = "recordings/Schwarm_einzel_schwer.mp4"
    cap = cv.VideoCapture(swarm_file)
    frame_by_frame = False
    previous_img = False

    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS))
    print(fps)

    # initialize the FourCC and a video writer object
    fourcc = cv.VideoWriter_fourcc("m", "p", "4", "v")
    output = cv.VideoWriter(
        "Schwarm_einzel_jet_to_gray_real_2.mp4",
        fourcc,
        fps,
        (frame_width, frame_height),
    )

    model = initialize_model()
    frame_no = 0
    last_frame = None
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            print(frame_no)
            break

        # Filter duplicate frames --> distorts the time dimension
        # if last_frame is None:
        #     last_frame = frame
        #     continue
        # elif abs(np.mean(frame - last_frame)) < 0.1:
        #     last_frame = frame
        #     print("Skipped")
        #     continue

        # WORK ON THE FRAME
        gray = jet_to_gray(model, frame)
        gray = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        output.write(gray)
        # cv.imshow('frame', gray)
        frame_no += 1
        print(f"Processed {frame_no/(180*20)} % of video.")

    cap.release()
    output.release()
    cv.destroyAllWindows()
