"""Basic API tests. They use a tiny random model and a synthetic video,
so they run in seconds and need no dataset.

Run from the project root:
    pip install pytest httpx
    pytest -q
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import SimpleVideoCNN   # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Create a fake model folder and start the app with it."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    torch.save(SimpleVideoCNN(num_classes=2).state_dict(), model_dir / "best_video_model.pth")
    (model_dir / "class_names.json").write_text(json.dumps(["jump", "sit"]))

    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")

    from api.main import app
    with TestClient(app) as test_client:   # "with" runs the startup (model loading)
        yield test_client


def make_video(path: Path, num_frames: int = 40):
    """Write a small synthetic .avi video."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 120))
    for t in range(num_frames):
        frame = np.full((120, 160, 3), 60, np.uint8)
        cv2.circle(frame, (80, 30 + t), 15, (0, 255, 0), -1)
        writer.write(frame)
    writer.release()


def test_health(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "classes": ["jump", "sit"]}


def test_predict_returns_valid_probabilities(client, tmp_path):
    video_path = tmp_path / "clip.avi"
    make_video(video_path)

    with video_path.open("rb") as video:
        response = client.post("/api/v1/predict", files={"file": ("clip.avi", video, "video/x-msvideo")})

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in ("jump", "sit")
    assert set(body["probabilities"]) == {"jump", "sit"}
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)


def test_predict_works_for_very_short_video(client, tmp_path):
    video_path = tmp_path / "short.avi"
    make_video(video_path, num_frames=5)   # fewer frames than the sequence length

    with video_path.open("rb") as video:
        response = client.post("/api/v1/predict", files={"file": ("short.avi", video, "video/x-msvideo")})
    assert response.status_code == 200


def test_wrong_file_type_is_rejected(client):
    response = client.post("/api/v1/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_broken_video_is_rejected(client):
    response = client.post("/api/v1/predict", files={"file": ("fake.avi", b"not a real video", "video/x-msvideo")})
    assert response.status_code == 400


def test_too_large_file_is_rejected(client):
    big_file = b"0" * (2 * 1024 * 1024)   # 2 MB, limit is 1 MB in this test
    response = client.post("/api/v1/predict", files={"file": ("big.avi", big_file, "video/x-msvideo")})
    assert response.status_code == 413