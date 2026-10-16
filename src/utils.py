"""Shared paths, constants and small helper functions used by every script.

Keeping them in one place means that, for example, SEQUENCE_LENGTH or a folder
name is defined exactly once and cannot get out of sync between scripts.

Note: torch is imported lazily inside the functions that need it, so scripts
like download_data.py work even before PyTorch is installed.
"""

import json
import random
import shutil
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths (ROOT is found automatically from this file's location: src/utils.py)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
ARCHIVE_PATH = DATA_DIR / "hmdb51_org.rar"
MAIN_DIR = DATA_DIR / "main"                      # main archive is unpacked here
RAW_VIDEOS_DIR = DATA_DIR / "raw"                 # raw/<class>/<video>.avi
METADATA_PATH = DATA_DIR / "video_metadata.csv"
EXTRACTED_FRAMES_DIR = DATA_DIR / "extracted_frames"   # <class>/<video>/frame_0000.jpg
FRAME_SUMMARY_PATH = EXTRACTED_FRAMES_DIR / "frame_extraction_summary.json"

FINAL_DATA_DIR = DATA_DIR / "final_data"
SEQUENCES_DIR = FINAL_DATA_DIR / "sequences"
TRAIN_DIR = FINAL_DATA_DIR / "train"
VAL_DIR = FINAL_DATA_DIR / "val"
TEST_DIR = FINAL_DATA_DIR / "test"

OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
PLOTS_DIR = OUTPUTS_DIR / "plots"
MODEL_PATH = MODELS_DIR / "best_video_model.pth"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
HISTORY_PATH = OUTPUTS_DIR / "training_history.json"
TEST_RESULTS_PATH = OUTPUTS_DIR / "test_results.json"

# ---------------------------------------------------------------------------
# Data settings
# ---------------------------------------------------------------------------
CLASSES = ["jump", "sit"]
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}

SEQUENCE_LENGTH = 24      # frames per clip (T)
IMAGE_SIZE = 128          # frames are resized to IMAGE_SIZE x IMAGE_SIZE
MEAN = (0.5, 0.5, 0.5)    # normalization used for every channel
STD = (0.5, 0.5, 0.5)

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15          # test ratio is whatever is left (0.15)
SEED = 42


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int = SEED):
    """Make runs reproducible."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    """Return the best available device: mps (Apple GPU) > cuda > cpu."""
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def reset_directory(path: Path):
    """Remove an old directory and recreate it empty."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def list_video_paths(root_dir: Path = RAW_VIDEOS_DIR):
    """Return all videos as sorted paths. Expected layout: root_dir/<class>/<video>."""
    return sorted(
        path for path in root_dir.glob("*/*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def sample_frame_indices(total_frames: int, sequence_length: int = SEQUENCE_LENGTH):
    """Pick `sequence_length` evenly spaced frame indices.

    Every clip ends up with exactly the same number of frames. If a video is
    shorter than `sequence_length`, the last frame is repeated.
    """
    if total_frames <= 0:
        return [0] * sequence_length

    if total_frames < sequence_length:
        indices = list(range(total_frames))
        indices += [total_frames - 1] * (sequence_length - total_frames)
        return indices

    return np.linspace(0, total_frames - 1, sequence_length, dtype=int).tolist()


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
