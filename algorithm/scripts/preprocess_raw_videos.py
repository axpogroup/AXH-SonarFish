from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

from algorithm.settings import Settings


def main(settings: Settings):
    input_directory_path = Path(settings.input_directory)
    output_directory_path = Path(settings.output_directory)
    input_video_file_paths = list(input_directory_path.glob("**/*.mp4"))

    for file_path in tqdm(input_video_file_paths, desc="Processing videos"):
        output_file_path = output_directory_path / file_path.relative_to(input_directory_path)
        if output_file_path.exists() and not settings.overwrite_existing_files:
            print(f"Skipping {file_path} since it already exists")
            continue

        print(f"Processing {file_path} -> {output_file_path}")
        down_sample_frame_rate_of_video(
            input_file=file_path,
            output_file=output_file_path,
            fps_out=settings.target_fps,
        )


def down_sample_frame_rate_of_video(input_file: Path, output_file: Path, fps_out: int):
    vidcap = cv2.VideoCapture(input_file.__str__())
    assert vidcap.isOpened()
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps_in = vidcap.get(cv2.CAP_PROP_FPS)
    index_in = -1
    index_out = -1
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    video_writer = cv2.VideoWriter(
        output_file.__str__().replace("_labels", ""),
        fourcc=fourcc,
        fps=fps_in,
        frameSize=(int(width), int(height)),
    )
    fail_rate = 0
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            success = vidcap.grab()
            if success:
                index_in += 1
                fail_rate = 0
                out_due = int(index_in / fps_in * fps_out)
                if out_due > index_out:
                    success, frame = vidcap.retrieve()
                    if success:
                        index_out += 1
                        video_writer.write(frame)
                        pbar.update(1)
            fail_rate += 1
            if fail_rate > 30:
                break
    # Release everything if job is finished
    vidcap.release()
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    with open("settings/preprocessing_settings.yaml") as f:
        settings = yaml.load(f, Loader=yaml.SafeLoader)
    main(settings)
