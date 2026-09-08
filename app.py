from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import uuid
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse
import requests
import cv2
import base64

from utils.config_loader import Config
from utils.path_manager import PathManager
from utils.logger import get_logger
import os

logger = get_logger(__name__, log_filename="api.log")

app = FastAPI(title="FaceRecognitionSystem API")

CONFIG_PATH = "configs/config.yaml"

# When set to '1', the API will remove the processed `output_path`
# returned by the pipeline immediately after processing finishes.
# Default: enabled (1) to save disk space on constrained hosts.
DELETE_OUTPUT_AFTER_PROCESS = os.getenv("DELETE_OUTPUT_AFTER_PROCESS", "1") == "1"

# In-memory job store. Fine for a single instance / demo use.
# NOTE: if you scale to multiple Render instances, or need jobs to
# survive a restart, replace this with Redis or a database table.
JOBS: dict[str, dict] = {}
JOBS_FACE: dict[str, dict] = {}

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

        if DELETE_OUTPUT_AFTER_PROCESS and output_path:
            try:
                # remove output file to free disk
                if os.path.exists(output_path):
                    os.remove(output_path)
                    JOBS[job_id]["output_removed"] = True
            except Exception:
                logger.exception("Failed to remove output for job %s", job_id)

    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(exc)

    finally:
        # Clean up the downloaded video (keep only the processed output).
        shutil.rmtree(tmp_dir, ignore_errors=True)


class ProcessURLRequest(BaseModel):
    url: str


def _run_face_extraction_job(video_url: str, job_id: str):
    JOBS_FACE[job_id] = {"status": "downloading", "results": []}
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"facejob_{job_id}_"))
    try:
        video_path = _download_video(video_url, tmp_dir)
        JOBS_FACE[job_id]["status"] = "processing"

        # Lazy-load the detector so environments that don't install insightface
        # still can run other endpoints.
        from detection.face_detector import FaceDetector

        cfg = Config(CONFIG_PATH)
        paths = PathManager(root_dir=cfg.project.root_dir)
        detector = FaceDetector(cfg, paths)

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_idx = 0
        results = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = detector.detect(frame)
            for det in faces:
                crop = det.crop(frame)
                _, buf = cv2.imencode('.jpg', crop)
                b64_img = base64.b64encode(buf.tobytes()).decode('utf-8')
                timestamp_sec = frame_idx / fps
                results.append({
                    "image": f"data:image/jpeg;base64,{b64_img}",
                    "timestamp": round(timestamp_sec, 2),
                })

            frame_idx += 1

        cap.release()
        JOBS_FACE[job_id] = {"status": "done", "results": results}

    except Exception as exc:
        logger.exception("Face extraction job %s failed", job_id)
        JOBS_FACE[job_id] = {"status": "error", "message": str(exc)}

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/api/health")
def health():
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


@app.post("/process-url")
def process_url(req: ProcessURLRequest, background_tasks: BackgroundTasks):
    job_id = uuid.uuid4().hex
    JOBS_FACE[job_id] = {"status": "queued", "results": []}
    background_tasks.add_task(_run_face_extraction_job, req.url, job_id)
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def status(job_id: str):
    job = JOBS_FACE.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# Serve the built React frontend (if present) from frontend/dist. Placed
# after API route definitions so API endpoints keep precedence.
try:
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")
except Exception:
    # If the build directory doesn't exist in this environment, mounting
    # will be a no-op; the API still works.
    logger.info("frontend/dist not found; static mount skipped")
 
