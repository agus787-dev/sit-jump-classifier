---
title: Sit Jump API
emoji: 🏃
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Sit / Jump video recognition

Classify short videos as `jump` or `sit` using a small CNN (HMDB51 dataset).

Each clip is turned into 12 evenly spaced frames. The same CNN encodes every
frame, the features are averaged over time, and a small classifier predicts the class.

## Setup

```bash
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
brew install unar          # needed to unpack the .rar archive (macOS)
```

## Run everything

```bash
python run_all.py
python run_all.py --epochs 10 --batch_size 4      # options for training
python run_all.py --from-step prepare_sequences   # skip earlier steps
```

## Run step by step

| Step | Command | What it does |
|------|---------|--------------|
| 1 | `python src/download_data.py` | Download `hmdb51_org.rar` (~2 GB) |
| 2 | `python src/extract_videos.py` | Unpack `jump` and `sit` videos, write `video_metadata.csv` |
| 3 | `python src/extract_frames.py` | Save every frame of every video as `.jpg` |
| 4 | `python src/prepare_sequences.py` | Pick 12 frames per video, split train/val/test (by video) |
| 5 | `python src/train.py` | Train, save best model, class names and history |
| 6 | `python src/plot_history.py` | Save loss/accuracy curves |
| 7 | `python src/evaluate.py` | Evaluate on the test set |
| - | `python src/predict.py video.avi` | Predict one new video |

## Run the FastAPI service

The API uses the trained checkpoint in `outputs/models/`. Train the model first
if these files do not exist:

```bash
python src/train.py
```

Run locally from the project root:

```bash
pip install -r requirements.txt
PYTHONPATH=src uvicorn api:app --reload
```

Open `http://localhost:8000/docs` for the interactive API. Upload an `.avi`,
`.mp4`, `.mov` or `.mkv` file as the `file` form field to `POST /predict`.
`GET /health` reports service liveness and `GET /ready` checks if the model is
loaded. Uploads are limited to 100 MiB by default; set `MAX_UPLOAD_BYTES` to
change the limit.

Build and run with Docker (the model files are mounted separately and are not
copied into the image):

```bash
docker build -t sit-jump-api .
docker run --rm -p 8000:8000 \
  -v "$PWD/outputs:/app/outputs:ro" \
  sit-jump-api
```

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@test_videos/example.avi"
```

## Project structure

```
src/
  utils.py              paths, constants, seed, device, small helpers
  download_data.py      download the archive
  extract_videos.py     unpack archive + metadata CSV
  extract_frames.py     videos -> frames
  prepare_sequences.py  frames -> fixed-length sequences + split
  dataset.py            Dataset and DataLoaders
  model.py              FrameEncoder and SimpleVideoCNN
  train.py              training loop
  plot_history.py       training curves
  evaluate.py           test-set evaluation
  predict.py            single-video prediction
run_all.py              runs all steps in order
```

Settings such as `CLASSES`, `SEQUENCE_LENGTH` and `IMAGE_SIZE` live in `src/utils.py`.
Generated folders `data/` and `outputs/` are ignored by git.
