import numpy as np
import cv2 as cv

from matplotlib import colormaps # colormaps['jet'], colormaps['turbo']
from matplotlib.colors import LinearSegmentedColormap
from matplotlib._cm import _jet_data


# TODO continue here with this little refactoring
def map_RGB_to_GRAY(src, colormap=False):
    if colormap:
        out = convert_jet_to_grey(src)
    else:
        out = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    return out


def blur_filter(src, kernel_size=10):
    out = cv.blur(src, (kernel_size, kernel_size))
    return out


def convert_jet_to_grey(img):
    (height, width) = img.shape[:2]

    cm = LinearSegmentedColormap("jet", _jet_data, N=2 ** 8)
    # cm = colormaps['turbo'] swap with jet if you use turbo colormap instead

    cm._init()  # Must be called first. cm._lut data field created here

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    fm = cv.FlannBasedMatcher(index_params, search_params)

    # JET, BGR order, excluding special palette values (>= 256)
    fm.add(255 * np.float32([cm._lut[:256, (2, 1, 0)]]))  # jet
    fm.train()

    # look up all pixels
    query = img.reshape((-1, 3)).astype(np.float32)
    matches = fm.match(query)

    # statistics: `result` is palette indices ("grayscale image")
    output = np.uint16([m.trainIdx for m in matches]).reshape(height, width)
    result = np.where(output < 256, output, 0).astype(np.uint8)
    # dist = np.uint8([m.distance for m in matches]).reshape(height, width)

    return result  # , dist uncomment if you wish accuracy image


if __name__ == '__main__':
    recording_file = "recordings/Jet_to_gray.mp4"
    cap = cv.VideoCapture(recording_file)
    frame_by_frame = False
    previous_img = False

    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # WORK ON THE FRAME
        gray = map_RGB_to_GRAY(frame)
        gray = blur_filter(gray, kernel_size=12)

        # Differential
        if previous_img is None:
            previous_img = gray
        current_image = gray
        #
        diff = abs(current_image-previous_img) + 125

        # Filter duplicate frames
        if abs(np.mean(diff)-125) < 0.1:
            continue

        # Threshold diffs
        diff[diff < 135] = 125
        # # quit()
        # gray = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        # output.write(gray)

        # out = np.concatenate((diff, frame), axis=1)
        cv.imshow('frame', diff)

        previous_img = gray

        # Play the video file
        if not frame_by_frame:
            usr_input = cv.waitKey(1)
        if usr_input == ord(' '):
            if cv.waitKey(0) == ord(' '):
                frame_by_frame = True
            else:
                frame_by_frame = False
            print("Press any key to continue ... ")
        if usr_input == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
