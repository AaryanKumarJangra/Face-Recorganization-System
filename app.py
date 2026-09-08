from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

import uuid
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse
import requests

from utils.config_loader import Config
from utils.path_manager import PathManager
from utils.logger import get_logger

logger = get_logger(__name__, log_filename="api.log")

app = FastAPI(title="FaceRecognitionSystem API")

CONFIG_PATH = "configs/config.yaml"

# In-memory job store. Fine for a single instance / demo use.
# NOTE: if you scale to multiple Render instances, or need jobs to
# survive a restart, replace this with Redis or a database table.
JOBS: dict[str, dict] = {}

# Loaded once at startup, reused across requests (matches how main.py's
# batch mode avoids reloading the detector/classifier per video).
_cfg = None
_paths = None
_recognizer = None


def get_recognizer():
    """Lazily create a single shared FaceRecognizer instance."""
    global _cfg, _paths, _recognizer
    if _recognizer is None:
        from recognition.recognize import FaceRecognizer
        _cfg = Config(CONFIG_PATH)
        _paths = PathManager(root_dir=_cfg.project.root_dir)
        _recognizer = FaceRecognizer(_cfg, _paths)
        logger.info("FaceRecognizer loaded and cached for API use.")
    return _recognizer


class RecognizeRequest(BaseModel):
    video_url: str


def _download_video(video_url: str, dest_dir: Path) -> Path:
    """
    Download a video from a direct URL to a local temp file, preserving
    the original filename when possible (video_timestamp.py relies on
    the CAMID_YYYYMMDD_HHMMSS naming pattern for timestamp parsing).
    """
    parsed = urlparse(video_url)
    filename = Path(parsed.path).name or f"{uuid.uuid4().hex}.mp4"
    dest_path = dest_dir / filename

    with requests.get(video_url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(r.raw, f)

    return dest_path


def _run_recognition_job(job_id: str, video_url: str):
    """Background worker: download the video, run the existing pipeline."""
    JOBS[job_id]["status"] = "downloading"
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"job_{job_id}_"))

    try:
        video_path = _download_video(video_url, tmp_dir)
        JOBS[job_id]["status"] = "processing"

        recognizer = get_recognizer()
        output_path = recognizer.process_video(str(video_path))

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["output_path"] = output_path

    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(exc)

    finally:
        # Clean up the downloaded video (keep only the processed output).
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/")
def health_check():
    return {"status": "running"}


@app.post("/recognize")
def recognize(request: RecognizeRequest, background_tasks: BackgroundTasks):
    """Start a recognition job for a video given by URL. Returns a job_id
    to poll via GET /jobs/{job_id}."""
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "queued", "video_url": request.video_url}

    background_tasks.add_task(_run_recognition_job, job_id, request.video_url)

    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
 
