"""
path_manager.py
================

Centralized path resolution for the entire FaceRecognitionSystem project.

Why this exists
---------------
The project requirements state "No hardcoded paths." In practice this means
no module should ever write something like:

    cv2.imwrite("dataset/person_001/img.jpg", frame)

Instead, every module asks a single `PathManager` instance for the path it
needs. This gives us three benefits:

1. The project can be moved/deployed anywhere without touching module code.
2. All paths are guaranteed to exist (this class creates them on first use).
3. If the folder structure ever changes, we only edit this one file.

Usage
-----
    from utils.path_manager import PathManager

    paths = PathManager(root_dir=".")
    person_dir = paths.get_person_dir("person_001")   # creates + returns Path
    db_path = paths.database_file()
"""

from __future__ import annotations

from pathlib import Path


class PathManager:
    """
    Resolves and (when needed) creates all project directories/files from a
    single root directory, keeping the rest of the codebase free of literal
    path strings.

    Parameters
    ----------
    root_dir : str
        Root directory of the project (typically the value of
        `project.root_dir` in configs/config.yaml).
    """

    def __init__(self, root_dir: str = ".") -> None:
        self.root: Path = Path(root_dir).resolve()

        # Top-level directories, declared once.
        self.dataset_dir: Path = self.root / "dataset"
        self.unknown_dir: Path = self.dataset_dir / "unknown"
        self.videos_dir: Path = self.root / "videos"
        self.output_dir: Path = self.root / "output"

        self.models_dir: Path = self.root / "models"
        self.models_detection_dir: Path = self.models_dir / "detection"
        self.models_recognition_dir: Path = self.models_dir / "recognition"
        self.models_tracking_dir: Path = self.models_dir / "tracking"
        self.models_bytetrack_dir: Path = self.models_tracking_dir / "bytetrack"

        self.database_dir: Path = self.root / "database"
        self.configs_dir: Path = self.root / "configs"
        self.training_dir: Path = self.root / "training"
        self.detection_dir: Path = self.root / "detection"
        self.tracking_dir: Path = self.root / "tracking"
        self.utils_dir: Path = self.root / "utils"
        self.logs_dir: Path = self.root / "logs"

        self._ensure_all_exist()

    def _ensure_all_exist(self) -> None:
        """Create every top-level directory if it doesn't already exist."""
        for directory in (
            self.dataset_dir,
            self.unknown_dir,
            self.videos_dir,
            self.output_dir,
            self.models_dir,
            self.models_detection_dir,
            self.models_recognition_dir,
            self.models_tracking_dir,
            self.models_bytetrack_dir,
            self.database_dir,
            self.configs_dir,
            self.training_dir,
            self.detection_dir,
            self.tracking_dir,
            self.utils_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset paths (Phase 3)
    # ------------------------------------------------------------------
    def get_person_dir(self, person_id: str) -> Path:
        """
        Return the dataset directory for a given person, creating it if
        necessary. E.g. get_person_dir("person_0001") ->
        <root>/dataset/person_0001/
        """
        person_dir = self.dataset_dir / person_id
        person_dir.mkdir(parents=True, exist_ok=True)
        return person_dir

    def get_unknown_dir(self) -> Path:
        """Return dataset/unknown/ — for face crops not yet assigned an identity."""
        self.unknown_dir.mkdir(parents=True, exist_ok=True)
        return self.unknown_dir

    # ------------------------------------------------------------------
    # Model paths — models/detection, models/recognition, models/tracking
    # ------------------------------------------------------------------
    def detection_model_file(self, filename: str) -> Path:
        """E.g. detection_model_file('retinaface.onnx') -> models/detection/retinaface.onnx"""
        return self.models_detection_dir / filename

    def recognition_model_file(self, filename: str) -> Path:
        """E.g. recognition_model_file('arcface.onnx') -> models/recognition/arcface.onnx"""
        return self.models_recognition_dir / filename

    def tracking_model_file(self, filename: str) -> Path:
        """E.g. tracking_model_file('bytetrack.yaml') -> models/tracking/bytetrack.yaml"""
        return self.models_tracking_dir / filename

    def model_file(self, filename: str) -> Path:
        """Generic fallback: a path directly inside models/."""
        return self.models_dir / filename

    # ------------------------------------------------------------------
    # Database paths — database/faces.db, embeddings.npy, metadata.json,
    # recognition_logs.db
    # ------------------------------------------------------------------
    def faces_db_file(self) -> Path:
        """SQLite DB holding per-face-crop metadata: database/faces.db"""
        return self.database_dir / "faces.db"

    def recognition_logs_db_file(self) -> Path:
        """SQLite DB holding per-frame recognition events: database/recognition_logs.db"""
        return self.database_dir / "recognition_logs.db"

    def embeddings_npy_file(self) -> Path:
        """NumPy array of all gallery embeddings: database/embeddings.npy"""
        return self.database_dir / "embeddings.npy"

    def metadata_json_file(self) -> Path:
        """Run/dataset summary metadata: database/metadata.json"""
        return self.database_dir / "metadata.json"

    def database_file(self, filename: str) -> Path:
        """Generic fallback: a path directly inside database/."""
        return self.database_dir / filename

    # ------------------------------------------------------------------
    # Output paths (Phase 7 — annotated/recognized videos)
    # ------------------------------------------------------------------
    def output_file(self, filename: str) -> Path:
        """Return a path inside output/, e.g. output_file('recognized_video.mp4')."""
        return self.output_dir / filename

    # ------------------------------------------------------------------
    # Log paths
    # ------------------------------------------------------------------
    def log_file(self, filename: str) -> Path:
        """Return a path inside logs/, e.g. log_file('training.log')."""
        return self.logs_dir / filename

    # ------------------------------------------------------------------
    # Video input paths
    # ------------------------------------------------------------------
    def video_file(self, filename: str) -> Path:
        """Return a path inside videos/ for a given input video filename."""
        return self.videos_dir / filename

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return f"PathManager(root={self.root})"
