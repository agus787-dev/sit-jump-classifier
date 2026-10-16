"""Inference code: video file -> prediction.

This module does NOT know anything about FastAPI, so it can be tested
(and reused) on its own.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# The ML code lives in src/, so we add it to the import path.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import get_frame_transform      # noqa: E402  (same transform as in training)
from model import SimpleVideoCNN             # noqa: E402
from utils import SEQUENCE_LENGTH, sample_frame_indices   # noqa: E402

MODEL_FILE_NAME = "best_video_model.pth"
CLASS_NAMES_FILE_NAME = "class_names.json"


class InvalidVideoError(ValueError):
    """Raised when the uploaded file cannot be read as a video."""


# ---------------------------------------------------------------------------
# Reading frames from a video
# ---------------------------------------------------------------------------
def count_frames(video_path: Path) -> int:
    """Return the number of frames in the video.

    Some formats (for example .webm recorded in a browser) do not store the
    frame count, so in that case we count the frames by reading through the file.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise InvalidVideoError("Could not open the video file.")

    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 0
        while capture.grab():
            total += 1
    capture.release()
    return total


def read_selected_frames(video_path: Path, indices) -> list:
    """Read only the frames at `indices` (BGR images), keeping memory usage small.

    The returned list has the same length and order as `indices`.
    If a frame cannot be read (the frame count in the file header can be wrong),
    the previous successfully read frame is used instead.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise InvalidVideoError("Could not open the video file.")

    wanted = set(indices)
    found = {}
    for frame_index in range(max(indices) + 1):
        if not capture.grab():          # grab() moves forward without decoding to an image
            break
        if frame_index in wanted:
            success, frame = capture.retrieve()
            if success:
                found[frame_index] = frame
    capture.release()

    if not found:
        raise InvalidVideoError("No frames could be read from the video.")

    frames, last_frame = [], None
    for index in indices:
        last_frame = found.get(index, last_frame)
        if last_frame is None:           # the very first wanted frame is missing
            last_frame = next(iter(found.values()))
        frames.append(last_frame)
    return frames


def to_training_like_image(frame_bgr: np.ndarray) -> Image.Image:
    """Make a frame look exactly like the frames the model was trained on.

    During training every frame was saved as a .jpg and loaded again, so we do
    the same JPEG round trip here. Then BGR (OpenCV) is converted to RGB (PIL).
    """
    success, buffer = cv2.imencode(".jpg", frame_bgr)
    if not success:
        raise InvalidVideoError("Could not process a video frame.")
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return Image.fromarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))


def video_to_tensor(video_path: Path) -> torch.Tensor:
    """Turn a video into one clip tensor of shape [1, T, C, H, W]."""
    total_frames = count_frames(video_path)
    if total_frames <= 0:
        raise InvalidVideoError("The video has no frames.")

    indices = sample_frame_indices(total_frames, SEQUENCE_LENGTH)
    frames = read_selected_frames(video_path, indices)

    transform = get_frame_transform()
    tensors = [transform(to_training_like_image(frame)) for frame in frames]
    return torch.stack(tensors, dim=0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------
class VideoPredictor:
    """Loads the trained model once and predicts the class of videos."""

    def __init__(self, model_dir: Path, device: str = "cpu"):
        model_dir = Path(model_dir)
        model_path = model_dir / MODEL_FILE_NAME
        class_names_path = model_dir / CLASS_NAMES_FILE_NAME

        for path in (model_path, class_names_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing model file: {path}")

        self.device = torch.device(device)
        self.class_names = json.loads(class_names_path.read_text(encoding="utf-8"))

        self.model = SimpleVideoCNN(num_classes=len(self.class_names)).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, video_path: Path) -> dict:
        """Return the predicted class and the probability of every class."""
        clip = video_to_tensor(video_path).to(self.device)

        with torch.no_grad():
            probabilities = torch.softmax(self.model(clip), dim=1)[0].cpu().tolist()

        best_index = int(np.argmax(probabilities))
        return {
            "prediction": self.class_names[best_index],
            "confidence": probabilities[best_index],
            "probabilities": dict(zip(self.class_names, probabilities)),
        }