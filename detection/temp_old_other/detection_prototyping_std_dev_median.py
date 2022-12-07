import cv2 as cv
import numpy as np
from matplotlib._cm import _jet_data
from matplotlib.colors import LinearSegmentedColormap

fish_area_mask = cv.imread('masks/fish.png', cv.IMREAD_GRAYSCALE)
full_area_mask = cv.imread('masks/full.png', cv.IMREAD_GRAYSCALE)

def map_RGB_to_GRAY(src, colormap=False):
    if colormap:
        out = convert_jet_to_grey(src)
    else:
        out = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    return out


def blur_filter(src, kernel_size=10):
    out = cv.blur(src, (kernel_size, kernel_size))
    return out


def mask_regions(img, area='fish'):
    masked = img
    if area == 'fish':
        np.place(masked, fish_area_mask < 100, 0)
    elif area == 'full':
        np.place(masked, full_area_mask < 100, 0)
    return masked

def convert_jet_to_grey(img):
    (height, width) = img.shape[:2]

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

    # look up all pixels
    query = img.reshape((-1, 3)).astype(np.float32)
    matches = fm.match(query)

    # statistics: `result` is palette indices ("grayscale image")
    output = np.uint16([m.trainIdx for m in matches]).reshape(height, width)
    result = np.where(output < 256, output, 0).astype(np.uint8)
    # dist = np.uint8([m.distance for m in matches]).reshape(height, width)

    return result  # , dist uncomment if you wish accuracy image


if __name__ == "__main__":
    recording_file = "../recordings/Schwarm_einzel_jet_to_gray_real.mp4"
    # recording_file = "recordings/new_settings/22-11-14_start_15-21-23_crop.mp4"
    output_file = "../output/normed_120_10_std_dev_threshold_2_median_11_schwarm_temp.mp4"
    # output_file = "output/normed_120.mp4"
    # output_file = "output/normed_120_minus_10.mp4"
    # output_file = "output/normed_120_10_std_dev_threshold_2.mp4"

    cap = cv.VideoCapture(recording_file)
    frame_by_frame = False
    previous_img = False

    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))*2
    fps = int(cap.get(cv.CAP_PROP_FPS))
    print(fps)

    # initialize the FourCC and a video writer object
    fourcc = cv.VideoWriter_fourcc("m", "p", "4", "v")
    output = cv.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    buffer = None
    last_frame = None
    frame_no = 0
    frames_total = 20*3*60 + 10*20
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # WORK ON THE INDIVIDUAL FRAME
        gray = map_RGB_to_GRAY(frame)
        gray = mask_regions(gray, area='full')
        # gray = blur_filter(gray, kernel_size=5)
        # median = cv.medianBlur(gray, 11)

        # Filter duplicate frames --> messes with the time dimension
        frame_no += 1
        # if last_frame is None:
        #     last_frame = gray
        #     continue
        # elif abs(np.mean(gray - last_frame)) < 25:
        #     last_frame = gray
        #     print("Dropped frame.")
        #     continue
        # print("Difference between two frames ", abs(np.mean(gray - last_frame)))
        last_frame = gray

        # FRAME MERGING
        if buffer is None:
            buffer = gray[:, :, np.newaxis]
        else:
            buffer = np.concatenate((gray[..., np.newaxis], buffer), axis=2)
        max_frames = 120
        if buffer.shape[2] > max_frames:
            buffer = buffer[:, :, :max_frames]
        elif buffer.shape[2] < max_frames:
            # Only do calculations once there is enough data
            cv.imshow("frame", gray)
            continue

        # current_avg = np.mean(buffer[:, :, :15], axis=2).astype('uint8')
        # noise_avg = np.mean(buffer[:, :, 15:], axis=2).astype('uint8')
        mean_img = np.mean(buffer[:, :, :], axis=2).astype('uint8')
        current_img = np.mean(buffer[:, :, :10], axis=2).astype('uint8')
        std_dev = np.std(buffer, axis=2).astype('uint8')

        normed = abs(current_img - mean_img) + 125
        # Threshold normed image - change has to be above a certain value / relative to std-dev of that pixel?
        normed[abs(normed.astype('int8')-125) < 2*std_dev] = 125

        # Filter
        # std_dev[std_dev < 20] = 0
        normed = cv.medianBlur(normed, 11)

        # Differential
        # if previous_img is None:
        #     previous_img = avg_30
        # current_image = avg_30

        # diff_15_30 = (current_avg - noise_avg) + 125
        # diff = abs(current_image - previous_img) + 125

        # # Filter duplicate frames
        # if abs(np.mean(diff) - 125) < 0.1:
        #     continue

        # # Threshold diffs
        # diff[diff < 135] = 125

        # OUTPUT
        out = normed
        disp = np.concatenate((out, gray))

        cv.imshow("frame", disp)
        # previous_img = avg_30

        # Write a video
        disp = cv.cvtColor(disp, cv.COLOR_GRAY2BGR)
        output.write(disp)

        print(f"Processed {frame_no/frames_total*100} % of video.")

        # Play the video file
        if not frame_by_frame:
            usr_input = cv.waitKey(1)
        if usr_input == ord(" "):
            if cv.waitKey(0) == ord(" "):
                frame_by_frame = True
            else:
                frame_by_frame = False
            print("Press any key to continue ... ")
        if usr_input == ord("q"):
            break

    cap.release()
    output.release()
    cv.destroyAllWindows()
