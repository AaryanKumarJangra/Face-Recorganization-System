"""
utils/database.py
==================

SQLite persistence layer for two databases:

    database/faces.db             — one row per SAVED face crop image
    database/recognition_logs.db  — one row per recognition event (Phase 7)

Kept as plain sqlite3 (stdlib, no extra dependency) with explicit schemas
and parameterized queries throughout (no string-formatted SQL — avoids
injection issues even though inputs here are trusted/internal).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__, log_filename="database.log")


class FacesDatabase:
    """
    Manages database/faces.db — metadata for every saved face crop.

    Schema
    ------
    faces(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id TEXT NOT NULL,
        track_id INTEGER NOT NULL,
        image_path TEXT NOT NULL,
        source_video TEXT,
        frame_number INTEGER,
        confidence REAL,
        embedding BLOB,
        quality_score REAL,
        is_blurry INTEGER,
        laplacian_var REAL,
        brightness_ok INTEGER,
        mean_brightness REAL,
        pose_ok INTEGER,
        yaw REAL,
        pitch REAL,
        is_low_res INTEGER,
        width INTEGER,
        height INTEGER,
        is_best_face INTEGER DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )

    NOTE on image_path: no longer UNIQUE. In best_face_only mode (see
    config.yaml dataset.best_face_only), every frame observation of a
    person is still logged here (with its own embedding, for training),
    but only ONE physical image file is ever written per person —
    image_path for all of that person's rows points at the same file,
    which gets overwritten whenever a higher-quality frame is seen.
    `is_best_face` marks which row was the one actually saved to disk.

    The `embedding` column stores the 512-d ArcFace embedding computed at
    COLLECTION time (i.e. from the original full video frame, before
    cropping) as raw float32 bytes. This is important: re-running face
    detection on an already-cropped face image is unreliable (RetinaFace
    relies on scene-level context that a tight crop doesn't have), so
    every downstream step that needs embeddings (training, consolidation)
    reads them from here instead of re-detecting on saved crop images.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    image_path TEXT,
                    image_blob BLOB,
                    source_video TEXT,
                    frame_number INTEGER,
                    confidence REAL,
                    embedding BLOB,
                    landmarks BLOB,
                    quality_score REAL,
                    is_blurry INTEGER,
                    laplacian_var REAL,
                    brightness_ok INTEGER,
                    mean_brightness REAL,
                    pose_ok INTEGER,
                    yaw REAL,
                    pitch REAL,
                    is_low_res INTEGER,
                    width INTEGER,
                    height INTEGER,
                    is_best_face INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # Ensure legacy DBs get the new image_blob column if missing
            cols = [r[1] for r in conn.execute("PRAGMA table_info(faces)").fetchall()]
            if "image_blob" not in cols:
                try:
                    conn.execute("ALTER TABLE faces ADD COLUMN image_blob BLOB")
                except sqlite3.OperationalError:
                    # If table doesn't exist yet or column already present, ignore
                    pass
            conn.execute("CREATE INDEX IF NOT EXISTS idx_faces_person_id ON faces(person_id)")

    def insert_face(
        self,
        person_id: str,
        track_id: int,
        image_path: str | None,
        image_blob: bytes | None,
        source_video: str,
        frame_number: int,
        confidence: float,
        embedding: "np.ndarray | None",
        quality_score: float,
        is_blurry: bool,
        laplacian_var: float,
        brightness_ok: bool,
        mean_brightness: float,
        pose_ok: bool,
        yaw: float,
        pitch: float,
        is_low_res: bool,
        width: int,
        height: int,
        is_best_face: bool = False,
        landmarks: "np.ndarray | None" = None,
    ) -> int:
        """Insert one face OBSERVATION record (not necessarily saved to disk). Returns the new row id."""
        landmarks_blob = landmarks.astype("float32").tobytes() if landmarks is not None else None
        embedding_blob = embedding.astype("float32").tobytes() if embedding is not None else None
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO faces (
                    person_id, track_id, image_path, source_video, frame_number,
                    confidence, embedding, landmarks, quality_score, is_blurry, laplacian_var, brightness_ok,
                    mean_brightness, pose_ok, yaw, pitch, is_low_res, width, height, is_best_face, image_blob
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    person_id,
                    track_id,
                    image_path or None,
                    source_video,
                    frame_number,
                    confidence,
                    embedding_blob,
                    landmarks_blob,
                    quality_score,
                    int(is_blurry),
                    laplacian_var,
                    int(brightness_ok),
                    mean_brightness,
                    int(pose_ok),
                    yaw,
                    pitch,
                    int(is_low_res),
                    width,
                    height,
                    int(is_best_face),
                    image_blob,
                ),
            )
            row_id = cursor.lastrowid
            # If an image blob was provided but no explicit image_path, point
            # image_path at the DB row (db://<id>) for callers that expect a path string.
            if image_blob is not None and not image_path:
                conn.execute("UPDATE faces SET image_path = ? WHERE id = ?", (f"db://{row_id}", row_id))
            return row_id

    def clear_best_face_flag(self, person_id: str) -> None:
        """Unmark any previous best-face row for this person (used when a better one is found)."""
        with self._connect() as conn:
            conn.execute("UPDATE faces SET is_best_face = 0 WHERE person_id = ?", (person_id,))

    def mark_best_face(self, row_id: int, image_path: str) -> None:
        """Mark a specific row as the current best face and update its saved image_path."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE faces SET is_best_face = 1, image_path = ? WHERE id = ?",
                (image_path, row_id),
            )

    def get_image_blob(self, row_id: int) -> Optional[bytes]:
        """Retrieve the stored image blob for a given row id, or None."""
        with self._connect() as conn:
            row = conn.execute("SELECT image_blob FROM faces WHERE id = ?", (row_id,)).fetchone()
            return row["image_blob"] if row is not None else None

    def get_best_faces(self):
        """Returns rows marked as the saved best face for each person."""
        with self._connect() as conn:
            return conn.execute(
                "SELECT id, person_id, image_path, landmarks, quality_score FROM faces WHERE is_best_face = 1"
            ).fetchall()

    def get_all_embeddings(self):
        """
        Returns (image_paths, person_ids, embeddings) for every row that has
        a stored embedding — used by training and consolidation instead of
        re-running face detection on saved crop images.
        """
        import numpy as np

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT image_path, person_id, embedding FROM faces WHERE embedding IS NOT NULL"
            ).fetchall()

        image_paths, person_ids, embeddings = [], [], []
        for row in rows:
            image_paths.append(row["image_path"])
            person_ids.append(row["person_id"])
            embeddings.append(np.frombuffer(row["embedding"], dtype=np.float32))

        return image_paths, person_ids, np.array(embeddings, dtype=np.float32) if embeddings else np.empty((0, 512), dtype=np.float32)

    def get_all_embeddings_full(self):
        """
        Like get_all_embeddings, but also returns row `id` and
        `quality_score` for each observation — needed by the consolidator
        to know which single row should become the new best face after
        merging person folders.

        Returns
        -------
        list[sqlite3.Row] — each with .id, .person_id, .image_path,
        .quality_score, .embedding (raw bytes — caller decodes with
        np.frombuffer)
        """
        with self._connect() as conn:
            return conn.execute(
                "SELECT id, person_id, image_path, quality_score, embedding "
                "FROM faces WHERE embedding IS NOT NULL"
            ).fetchall()

    def reassign_rows(self, row_ids: List[int], new_person_id: str, new_image_path: str, best_row_id: Optional[int] = None) -> None:
        """
        Bulk-update a set of rows to a new (merged) person_id and shared
        image_path, and mark exactly one of them (best_row_id) as the
        saved best face. Used by the consolidator after clustering.
        """
        with self._connect() as conn:
            placeholders = ",".join("?" for _ in row_ids)
            conn.execute(
                f"UPDATE faces SET person_id = ?, image_path = ?, is_best_face = 0 "
                f"WHERE id IN ({placeholders})",
                (new_person_id, new_image_path, *row_ids),
            )
            if best_row_id is not None:
                conn.execute("UPDATE faces SET is_best_face = 1 WHERE id = ?", (best_row_id,))

    def count_for_person(self, person_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as c FROM faces WHERE person_id = ?", (person_id,)
            ).fetchone()
            return row["c"]

    def get_all_persons(self) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT DISTINCT person_id FROM faces ORDER BY person_id").fetchall()
            return [r["person_id"] for r in rows]

    def get_faces_for_person(self, person_id: str) -> List[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM faces WHERE person_id = ? ORDER BY frame_number", (person_id,)
            ).fetchall()


class RecognitionLogsDatabase:
    """
    Manages database/recognition_logs.db — one row per recognition event,
    written during Phase 7 (recognizing faces in new videos).

    Schema
    ------
    recognition_logs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_source TEXT,
        frame_number INTEGER,
        track_id INTEGER,
        predicted_person_id TEXT,
        confidence REAL,
        is_unknown INTEGER,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recognition_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_source TEXT,
                    frame_number INTEGER,
                    track_id INTEGER,
                    predicted_person_id TEXT,
                    confidence REAL,
                    is_unknown INTEGER,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def log_event(
        self,
        video_source: str,
        frame_number: int,
        track_id: int,
        predicted_person_id: str,
        confidence: float,
        is_unknown: bool,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO recognition_logs (
                    video_source, frame_number, track_id,
                    predicted_person_id, confidence, is_unknown
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (video_source, frame_number, track_id, predicted_person_id, confidence, int(is_unknown)),
            )
