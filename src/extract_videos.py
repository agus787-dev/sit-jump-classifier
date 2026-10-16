"""Step 2: unpack the HMDB51 archive and write a metadata CSV for every video.

The main archive contains one .rar file per class (jump.rar, sit.rar, ...).
We only unpack the classes listed in CLASSES.

Requires the `unar` command line tool:  brew install unar

Usage:
    python src/extract_videos.py
"""

import csv
import shutil
import subprocess
from pathlib import Path

import cv2

from utils import (
    ARCHIVE_PATH,
    CLASSES,
    MAIN_DIR,
    METADATA_PATH,
    RAW_VIDEOS_DIR,
    list_video_paths,
)

METADATA_FIELDS = [
    "class", "file", "readable", "frames", "fps", "width", "height",
    "duration_seconds", "size_bytes", "error",
]


# ---------------------------------------------------------------------------
# Unpacking
# ---------------------------------------------------------------------------
def extract_rar(archive: Path, destination: Path):
    """Extract a .rar file using the `unar` command line tool."""
    if shutil.which("unar") is None:
        raise RuntimeError("`unar` is not installed. Install it with: brew install unar")

    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(["unar", "-f", "-D", "-o", str(destination), str(archive)], check=True)


def extract_class_archives():
    """Unpack the main archive, then only the class archives we need."""
    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(f"{ARCHIVE_PATH} not found. Run download_data.py first.")

    print("Extracting main archive...")
    extract_rar(ARCHIVE_PATH, MAIN_DIR)

    for class_name in CLASSES:
        class_archive = next(MAIN_DIR.rglob(f"{class_name}.rar"))
        print(f"Extracting class: {class_name}")
        extract_rar(class_archive, RAW_VIDEOS_DIR)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
def read_video_info(video_path: Path) -> dict:
    """Open one video with OpenCV and return one metadata row."""
    row = {
        "class": video_path.parent.name,
        "file": str(video_path),
        "readable": False,
        "frames": "",
        "fps": "",
        "width": "",
        "height": "",
        "duration_seconds": "",
        "size_bytes": video_path.stat().st_size,
        "error": "",
    }

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            row["error"] = "OpenCV could not open this video"
            return row

        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        row.update({
            "readable": True,
            "frames": frames,
            "fps": round(fps, 3),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration_seconds": round(frames / fps, 3) if fps > 0 else "",
        })
        return row
    finally:
        capture.release()


def print_video_info(row: dict):
    """Print one metadata row in a readable single line."""
    if not row["readable"]:
        print(f"{row['file']} | ERROR: {row['error']}")
        return

    size_mb = row["size_bytes"] / (1024 * 1024)
    print(
        f"{row['file']} | {row['frames']} frames | {row['fps']} FPS | "
        f"{row['width']}x{row['height']} | {row['duration_seconds']} s | {size_mb:.2f} MB"
    )


def write_metadata(rows):
    """Save all metadata rows to a CSV file."""
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METADATA_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    extract_class_archives()

    video_paths = list_video_paths()
    print(f"\nVideo metadata ({len(video_paths)} videos):")

    rows = []
    for video_path in video_paths:
        row = read_video_info(video_path)
        print_video_info(row)
        rows.append(row)

    write_metadata(rows)
    print(f"\nMetadata for all videos saved to: {METADATA_PATH}")


if __name__ == "__main__":
    main()
