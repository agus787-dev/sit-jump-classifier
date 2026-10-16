"""Step 3: save every frame of every video as a .jpg image.

Output layout:
    data/extracted_frames/<class>/<video_name>/frame_0000.jpg, frame_0001.jpg, ...

Usage:
    python src/extract_frames.py
"""

from pathlib import Path

import cv2

from utils import (
    EXTRACTED_FRAMES_DIR,
    FRAME_SUMMARY_PATH,
    RAW_VIDEOS_DIR,
    list_video_paths,
    reset_directory,
    save_json,
)


def extract_all_frames(video_path: Path, output_dir: Path):
    """Read every frame of one video and save it as an image.

    Returns the number of saved frames, or None if the video cannot be opened.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None

    saved_count = 0
    while True:
        success, frame = capture.read()
        if not success:
            break

        output_path = output_dir / f"frame_{saved_count:04d}.jpg"
        cv2.imwrite(str(output_path), frame)
        saved_count += 1

    capture.release()
    return saved_count


def process_video(video_path: Path):
    """Extract frames of one video into data/extracted_frames/<class>/<video>/."""
    class_name = video_path.parent.name
    output_dir = EXTRACTED_FRAMES_DIR / class_name / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return class_name, extract_all_frames(video_path, output_dir)


def main():
    video_paths = list_video_paths()
    if not video_paths:
        raise FileNotFoundError(
            f"No videos found in {RAW_VIDEOS_DIR}. Please run extract_videos.py first."
        )

    reset_directory(EXTRACTED_FRAMES_DIR)

    summary = {"frames_per_video": {}, "skipped_videos": []}

    for video_path in video_paths:
        class_name, saved_count = process_video(video_path)

        if saved_count is None:
            print(f"WARNING: could not open {video_path.name}, skipping it")
            summary["skipped_videos"].append(str(video_path))
            continue

        summary["frames_per_video"][f"{class_name}/{video_path.stem}"] = saved_count
        print(f"Extracted {saved_count} frames from {video_path.name}")

    save_json(summary, FRAME_SUMMARY_PATH)
    print(f"\nProcessed {len(summary['frames_per_video'])} videos, "
          f"skipped {len(summary['skipped_videos'])}.")
    print("Saved extraction summary to:", FRAME_SUMMARY_PATH)


if __name__ == "__main__":
    main()
