import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def process_video(input_filename: str, boosting_alpha: float = 2.0, boosting_beta: float = 30.0):
    assert Path(input_filename).exists(), f"File {input_filename} does not exist."
    cap = cv2.VideoCapture(input_filename)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_filename = input_filename.rsplit(".", 1)[0] + "_jet.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use 'mp4v' codec
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    for _ in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        fail_count = 0
        if not ret and fail_count < 5:
            fail_count += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.convertScaleAbs(gray, alpha=boosting_alpha, beta=boosting_beta)
        jet = plt.get_cmap("jet")(enhanced)
        jet_bgr = cv2.cvtColor((jet * 255).astype(np.uint8), cv2.COLOR_RGBA2BGR)
        out.write(jet_bgr)

    cap.release()
    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video file.")
    parser.add_argument("path", help="The path to the video file to process.")
    parser.add_argument("--boosting_alpha", type=float, default=2.0, help="The boosting alpha value.")
    parser.add_argument("--boosting_beta", type=float, default=30.0, help="The boosting beta value.")
    args = parser.parse_args()

    process_video(args.path, args.boosting_alpha, args.boosting_beta)
