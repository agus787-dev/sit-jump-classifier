"""Predict the class of ONE new video with the trained model.

Usage:
    python src/predict.py path/to/video.avi
"""

import argparse
from pathlib import Path

import cv2
import torch
from PIL import Image

from dataset import get_frame_transform
from evaluate import load_trained_model
from utils import (
    CLASS_NAMES_PATH,
    SEQUENCE_LENGTH,
    get_device,
    load_json,
    sample_frame_indices,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict the class of one video.")
    parser.add_argument("video", type=Path, help="Path to a video file.")
    return parser.parse_args()


def read_all_frames(video_path: Path):
    """Read every frame of the video as a list of RGB PIL images."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frames = []
    while True:
        success, frame = capture.read()
        if not success:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # OpenCV reads BGR
        frames.append(Image.fromarray(frame_rgb))
    capture.release()

    if not frames:
        raise ValueError(f"No frames could be read from: {video_path}")
    return frames


def video_to_sequence(video_path: Path) -> torch.Tensor:
    """Turn a video into a batch of one clip: shape [1, T, C, H, W]."""
    frames = read_all_frames(video_path)
    indices = sample_frame_indices(len(frames), SEQUENCE_LENGTH)

    transform = get_frame_transform()
    tensors = [transform(frames[index]) for index in indices]
    return torch.stack(tensors, dim=0).unsqueeze(0)


def main():
    args = parse_args()
    device = get_device()

    class_names = load_json(CLASS_NAMES_PATH)
    model = load_trained_model(class_names, device)

    sequence = video_to_sequence(args.video).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(sequence), dim=1)[0].cpu()

    best_index = int(torch.argmax(probabilities))
    print(f"\nPrediction: {class_names[best_index]} ({probabilities[best_index]:.1%})")
    for name, probability in zip(class_names, probabilities):
        print(f"  {name}: {probability:.1%}")


if __name__ == "__main__":
    main()
