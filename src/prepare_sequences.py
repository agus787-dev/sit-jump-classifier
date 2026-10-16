"""Step 4: turn extracted frames into fixed-length sequences and split them.

For every video:
    1. pick SEQUENCE_LENGTH evenly spaced frames  -> one sequence folder
    2. the whole video goes to train, val OR test (split by video, not by frame,
       so frames of one video never end up in two different splits)

Output layout:
    data/final_data/sequences/<class>/<video>/frame_00.jpg ... frame_11.jpg
    data/final_data/train|val|test/<class>/<video>/...

Usage:
    python src/prepare_sequences.py
"""

import random
import shutil
from pathlib import Path

from utils import (
    CLASSES,
    EXTRACTED_FRAMES_DIR,
    SEED,
    SEQUENCE_LENGTH,
    SEQUENCES_DIR,
    TEST_DIR,
    TRAIN_DIR,
    TRAIN_RATIO,
    VAL_DIR,
    VAL_RATIO,
    reset_directory,
    sample_frame_indices,
)


def copy_sequence_frames(source_frame_paths, destination_dir: Path):
    """Copy the selected frames into one sequence folder, keeping time order."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    for new_index, frame_path in enumerate(source_frame_paths):
        shutil.copy2(frame_path, destination_dir / f"frame_{new_index:02d}.jpg")


def create_sequence(video_frames_dir: Path, class_name: str) -> Path:
    """Build one sequence folder from all frames of one video."""
    frame_paths = sorted(video_frames_dir.glob("*.jpg"))
    indices = sample_frame_indices(len(frame_paths), SEQUENCE_LENGTH)
    selected_frames = [frame_paths[index] for index in indices]

    sequence_dir = SEQUENCES_DIR / class_name / video_frames_dir.name
    copy_sequence_frames(selected_frames, sequence_dir)
    return sequence_dir


def split_list(items, train_ratio: float = TRAIN_RATIO, val_ratio: float = VAL_RATIO):
    """Split a list into train / val / test parts (test gets the remainder)."""
    if len(items) < 3:
        raise ValueError(f"Need at least 3 videos per class to split, got {len(items)}.")

    train_count = int(len(items) * train_ratio)
    val_count = int(len(items) * val_ratio)

    train_items = items[:train_count]
    val_items = items[train_count:train_count + val_count]
    test_items = items[train_count + val_count:]
    return train_items, val_items, test_items


def copy_sequences_to_split(sequence_dirs, split_dir: Path, class_name: str):
    """Copy sequence folders into train/, val/ or test/."""
    for sequence_dir in sequence_dirs:
        shutil.copytree(sequence_dir, split_dir / class_name / sequence_dir.name)


def count_items_per_class(root_dir: Path) -> dict:
    """Count how many sequence folders each class has inside root_dir."""
    counts = {}
    if not root_dir.exists():
        return counts
    for class_dir in sorted(root_dir.iterdir()):
        if class_dir.is_dir():
            counts[class_dir.name] = sum(1 for item in class_dir.iterdir() if item.is_dir())
    return counts


def main():
    random.seed(SEED)

    if not EXTRACTED_FRAMES_DIR.exists():
        raise FileNotFoundError(
            f"No extracted frames found in {EXTRACTED_FRAMES_DIR}. Please run extract_frames.py first."
        )

    for folder in [SEQUENCES_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR]:
        reset_directory(folder)

    for class_name in CLASSES:
        video_dirs = sorted(p for p in (EXTRACTED_FRAMES_DIR / class_name).glob("*") if p.is_dir())
        random.shuffle(video_dirs)

        sequence_dirs = [create_sequence(video_dir, class_name) for video_dir in video_dirs]
        train_seqs, val_seqs, test_seqs = split_list(sequence_dirs)

        copy_sequences_to_split(train_seqs, TRAIN_DIR, class_name)
        copy_sequences_to_split(val_seqs, VAL_DIR, class_name)
        copy_sequences_to_split(test_seqs, TEST_DIR, class_name)

    print("Sequence preparation is complete.\n")
    print("Train sequence counts:", count_items_per_class(TRAIN_DIR))
    print("Val sequence counts:  ", count_items_per_class(VAL_DIR))
    print("Test sequence counts: ", count_items_per_class(TEST_DIR))


if __name__ == "__main__":
    main()
