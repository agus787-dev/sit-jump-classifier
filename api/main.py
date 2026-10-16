"""FastAPI app: upload a video, get back `jump` or `sit`.

Run locally (from the project root):
    uvicorn api.main:app --reload --port 8000

Then open http://localhost:8000/docs to try it in the browser.

Settings (environment variables, all optional):
    MODEL_DIR        folder with best_video_model.pth and class_names.json   (default: models/)
    ALLOWED_ORIGINS  comma separated frontend URLs for CORS                  (default: http://localhost:5173)
    MAX_UPLOAD_MB    maximum upload size in megabytes                        (default: 20)
    DEVICE           torch device, "cpu" or "cuda"                           (default: cpu)
"""

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .inference import InvalidVideoError, ROOT, VideoPredictor

ALLOWED_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv", ".webm"}
CHUNK_SIZE = 1024 * 1024   # read uploads 1 MB at a time


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
def get_model_dir() -> Path:
    return Path(os.getenv("MODEL_DIR", ROOT / "models"))


def get_allowed_origins() -> list:
    raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def get_max_upload_bytes() -> int:
    return int(float(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    classes: list[str]


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def validate_video_name(filename: str | None) -> str:
    """Check the file extension and return it (lowercase)."""
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_SUFFIXES))
        raise HTTPException(status_code=415, detail=f"Unsupported file type. Use one of: {allowed}")
    return suffix


def save_upload_to_temp_file(upload: UploadFile, suffix: str) -> Path:
    """Save the upload to a temporary file, stopping if it is too large."""
    max_bytes = get_max_upload_bytes()
    size = 0

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = Path(temp_file.name)
        try:
            while chunk := upload.file.read(CHUNK_SIZE):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File is too large. Maximum size is {max_bytes // (1024 * 1024)} MB.",
                    )
                temp_file.write(chunk)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    if size == 0:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    return temp_path


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the server starts (not on every request)."""
    app.state.predictor = VideoPredictor(get_model_dir(), device=os.getenv("DEVICE", "cpu"))
    yield


app = FastAPI(title="Sit / Jump video classifier", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

router = APIRouter(prefix="/api/v1")


@router.get("/health", response_model=HealthResponse)
def health():
    """Used to check that the server is alive and the model is loaded."""
    return {"status": "ok", "classes": app.state.predictor.class_names}


@router.post("/predict", response_model=PredictionResponse)
def predict(file: UploadFile = File(...)):
    """Upload a short video and get the predicted class with probabilities."""
    suffix = validate_video_name(file.filename)
    temp_path = save_upload_to_temp_file(file, suffix)

    try:
        return app.state.predictor.predict(temp_path)
    except InvalidVideoError as error:
        raise HTTPException(status_code=400, detail=str(error))
    finally:
        temp_path.unlink(missing_ok=True)   # never keep uploaded videos


app.include_router(router)


@app.get("/", include_in_schema=False)
def root():
    return {"message": "Sit/Jump video classifier API", "docs": "/docs"}
