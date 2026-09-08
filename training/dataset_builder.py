"""
training/dataset_builder.py
=============================

End-to-end pipeline for Phase 1-4 combined:
video -> detect (RetinaFace) -> track (ByteTrack + re-ID) -> save EVERY
face crop -> record metadata in faces.db.

Per your requirement, `dataset.save_all_faces: true` in config.yaml means
NOTHING is rejected here — blurry, dark, extreme side-angle, tiny faces
are all saved. Quality is still computed and stored as metadata (see
utils/quality.py) so you can filter later at training time if you want to.

Usage
-----
    python main.py --build-dataset --video videos/sample.mp4
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from detection.face_detector import FaceDetector
from tracking.face_tracker import FaceTracker
from utils.config_loader import Config
from utils.database import FacesDatabase
from utils.logger import get_logger
from utils.path_manager import PathManager
from utils.quality import assess_quality
from utils.video_timestamp import parse_video_timestamp

logger = get_logger(__name__, log_filename="dataset_builder.log")


class DatasetBuilder:
    """
    Orchestrates detection + tracking + saving for one input video.

    Parameters
    ----------
    config : Config
    paths : PathManager
    """

    def __init__(self, config: Config, paths: PathManager) -> None:
        self.config = config
        self.paths = paths

        logger.info("Initializing FaceDetector...")
        self.detector = FaceDetector(config, paths)

        # Tracker is (re)created per-video inside process_video() so that a
        # single DatasetBuilder instance can safely process many videos
        # back-to-back (bulk mode) without reloading the detector each time,
        # while still starting each video with a clean tracker and
        # continuing person_XXXX numbering across the whole batch.
        self.tracker = None

        self.faces_db = FacesDatabase(paths.faces_db_file())

        # Per-person running image counter, used only for filenames.
        self._image_counters: Dict[str, int] = {}
        self._new_person_created: Dict[str, bool] = {}
        # Tracks the current best composite quality score per person, used
        # in best_face_only mode to decide whether a new frame beats the
        # currently-saved best_face.jpg.
        self._best_face_scores: Dict[str, float] = {}
        # Actual file path currently on disk for each person's best face,
        # so we can remove the old timestamped file when a better one
        # replaces it.
        self._best_face_paths: Dict[str, Path] = {}

        # Parsed from the current video's filename (CAMID_YYYYMMDD_HHMMSS.ext).
        # None if the filename has no recognizable timestamp — callers fall
        # back to frame-number-based naming in that case.
        self._current_video_ts_info = None
        self._current_fps: float = 25.0

    @staticmethod
    def _determine_next_person_number(paths: PathManager) -> int:
        """
        Scan dataset/person_XXXX folders and return the next free number,
        so repeated --build-dataset runs (e.g. on multiple videos) don't
        collide different people into the same person_0001 folder.

        NOTE: this only prevents ID collisions across runs — it does NOT
        re-identify a person seen in video A as the same person in video B.
        Cross-video re-identification would require comparing against the
        full embeddings gallery (built by --train) rather than just the
        in-memory per-run gallery used during tracking.
        """
        existing = [
            d.name for d in paths.dataset_dir.iterdir()
            if d.is_dir() and d.name.startswith("person_")
        ]
        if not existing:
            return 1
        numbers = []
        for name in existing:
            try:
                numbers.append(int(name.split("_")[1]))
            except (IndexError, ValueError):
                continue
        return max(numbers, default=0) + 1

    def process_video(self, video_path: str) -> dict:
        """
        Run the full pipeline on one video. Returns a summary dict which
        is also written to database/metadata.json.
        """
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        # Fresh tracker for this video, numbering continued from whatever is
        # already on disk (so a bulk run over many videos never collides two
        # different people into the same person_XXXX folder).
        start_num = self._determine_next_person_number(self.paths)
        self.tracker = FaceTracker(self.config, start_person_num=start_num)

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        self._current_video_ts_info = parse_video_timestamp(video_path)
        self._current_fps = fps if fps and fps > 0 else 25.0
        if self._current_video_ts_info is None:
            logger.info(
                "No CAMID_YYYYMMDD_HHMMSS timestamp found in '%s' — "
                "saved filenames will use frame numbers instead.",
                video_path,
            )
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = max(1, self.config.performance.frame_skip)
        logger.info(
            "Processing video '%s' (%d frames, %.1f fps, detecting every %d frame(s))",
            video_path, total_frames, fps, frame_skip,
        )

        frame_number = 0
        saved_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # SPEED LEVER: only run the (expensive) detector every
            # `frame_skip` frames. At 15-30fps, a face moves very little
            # between consecutive frames, so skipping frames barely
            # affects tracking/best-face quality but directly divides
            # total processing time by frame_skip. Track aging
            # (tracking.track_buffer) is counted in PROCESSED frames, not
            # video frames, so behavior stays consistent regardless of
            # frame_skip — just document this if you tune track_buffer.
            if frame_number % frame_skip == 0:
                faces = self.detector.detect(frame)
                tracked_faces = self.tracker.update(faces)

                for tf in tracked_faces:
                    saved_count += self._save_face(tf, frame, video_path, frame_number)

            frame_number += 1
            if frame_number % 100 == 0:
                elapsed = time.time() - start_time
                proc_fps = frame_number / elapsed if elapsed > 0 else 0.0
                logger.info(
                    "Frame %d/%d | active tracks=%d | unique people=%d | %.1f fps",
                    frame_number, total_frames, self.tracker.active_track_count(),
                    self.tracker.total_unique_people(), proc_fps,
                )

        cap.release()
        elapsed = time.time() - start_time

        if self.config.dataset.best_face_only:
            self._build_bestfaces_gallery()

        summary = {
            "source_video": video_path,
            "total_frames_processed": frame_number,
            "total_faces_saved": saved_count,
            "unique_people_detected": self.tracker.total_unique_people(),
            "processing_time_seconds": round(elapsed, 2),
            "average_fps": round(frame_number / elapsed, 2) if elapsed > 0 else 0.0,
        }
        self._write_metadata_summary(summary)
        logger.info("Dataset build complete: %s", summary)
        return summary

    def _frame_timestamp_str(self, frame_number: int) -> Optional[str]:
        """Real-world 'YYYYMMDD_HHMMSS' for this frame, derived from the
        video filename's embedded start time + frame_number/fps. Returns
        None if the current video's filename had no recognizable timestamp."""
        if self._current_video_ts_info is None:
            return None
        return self._current_video_ts_info.frame_timestamp_str(frame_number, self._current_fps)

    def _save_face(self, tracked_face, frame: np.ndarray, video_path: str, frame_number: int) -> int:
        """
        Process one detected+tracked face for one frame.

        In best_face_only mode (default): every frame's observation is
        logged to faces.db (embedding + quality score included) so
        training still has multiple samples per person to learn from —
        but only ONE image file per person ever exists on disk, updated
        in place whenever a higher-quality frame is seen. Returns 1 if
        this observation became the new best (file written/overwritten),
        0 otherwise — used only for the "faces saved" summary count.

        In save_all_faces mode: behaves as before — every frame's crop is
        written as its own file.
        """
        from utils.quality import composite_quality_score

        df = tracked_face.detected_face
        person_id = tracked_face.person_id

        crop = df.crop(frame)
        if crop.size == 0:
            return 0

        crop_landmarks = df.crop_relative_landmarks(frame)
        quality = assess_quality(crop, df.yaw, df.pitch, self.config)
        score = composite_quality_score(quality, df.confidence, df.yaw, df.pitch)

        if self.config.dataset.best_face_only:
            return self._save_best_face_observation(
                tracked_face, crop, crop_landmarks, quality, score, video_path, frame_number
            )
        else:
            return self._save_every_frame(tracked_face, crop, crop_landmarks, quality, video_path, frame_number)

    def _save_best_face_observation(self, tracked_face, crop, crop_landmarks, quality, score, video_path, frame_number) -> int:
        person_id = tracked_face.person_id
        df = tracked_face.detected_face

        person_dir = self.paths.get_person_dir(person_id)
        ts_str = self._frame_timestamp_str(frame_number)
        best_image_name = f"best_face_{ts_str}.jpg" if ts_str else "best_face.jpg"
        best_image_path = person_dir / best_image_name

        current_best = self._best_face_scores.get(person_id)
        is_new_best = current_best is None or score > current_best

        image_path_to_log = str(best_image_path) if is_new_best else str(
            self._best_face_paths.get(person_id, best_image_path)
        )
        # Always log the observation (embedding) for training purposes,
        # even if it's not the new best — image_path just points at
        # wherever the current best file lives/will live.
        row_id = self.faces_db.insert_face(
            person_id=person_id,
            track_id=tracked_face.track_id,
            image_path=image_path_to_log,
            source_video=video_path,
            frame_number=frame_number,
            confidence=df.confidence,
            embedding=df.embedding,
            quality_score=score,
            is_blurry=quality.is_blurry,
            laplacian_var=quality.laplacian_var,
            brightness_ok=quality.brightness_ok,
            mean_brightness=quality.mean_brightness,
            pose_ok=quality.pose_ok,
            yaw=df.yaw,
            pitch=df.pitch,
            is_low_res=quality.is_low_res,
            width=quality.width,
            height=quality.height,
            is_best_face=is_new_best,
            landmarks=crop_landmarks,
        )

        if is_new_best:
            # Remove the previous best-face file (it had the OLD frame's
            # timestamp baked into its name) before writing the new one.
            old_path = self._best_face_paths.get(person_id)
            if old_path is not None and old_path != best_image_path and old_path.exists():
                old_path.unlink()

            cv2.imwrite(str(best_image_path), crop)
            self.faces_db.clear_best_face_flag(person_id)
            self.faces_db.mark_best_face(row_id, str(best_image_path))
            self._best_face_scores[person_id] = score
            self._best_face_paths[person_id] = best_image_path
            return 1
        return 0

    def _save_every_frame(self, tracked_face, crop, crop_landmarks, quality, video_path, frame_number) -> int:
        from utils.quality import composite_quality_score

        df = tracked_face.detected_face
        person_id = tracked_face.person_id
        score = composite_quality_score(quality, df.confidence, df.yaw, df.pitch)

        person_dir = self.paths.get_person_dir(person_id)
        idx = self._image_counters.get(person_id, 0) + 1
        self._image_counters[person_id] = idx

        video_stem = Path(video_path).stem
        ts_str = self._frame_timestamp_str(frame_number)
        if ts_str:
            filename = f"{person_id}_{video_stem}_track{tracked_face.track_id}_{ts_str}_{idx:04d}.jpg"
        else:
            filename = f"{person_id}_{video_stem}_track{tracked_face.track_id}_frame{frame_number:06d}_{idx:04d}.jpg"
        image_path = person_dir / filename

        suffix = 0
        final_path = image_path
        while final_path.exists():
            suffix += 1
            final_path = image_path.with_name(f"{image_path.stem}_{suffix}{image_path.suffix}")
        image_path = final_path

        cv2.imwrite(str(image_path), crop)

        self.faces_db.insert_face(
            person_id=person_id,
            track_id=tracked_face.track_id,
            image_path=str(image_path),
            source_video=video_path,
            frame_number=frame_number,
            confidence=df.confidence,
            embedding=df.embedding,
            quality_score=score,
            is_blurry=quality.is_blurry,
            laplacian_var=quality.laplacian_var,
            brightness_ok=quality.brightness_ok,
            mean_brightness=quality.mean_brightness,
            pose_ok=quality.pose_ok,
            yaw=df.yaw,
            pitch=df.pitch,
            is_low_res=quality.is_low_res,
            width=quality.width,
            height=quality.height,
            is_best_face=False,
            landmarks=crop_landmarks,
        )
        return 1

    def _build_bestfaces_gallery(self) -> None:
        """
        Copies every person's best_face.jpg into a single flat
        dataset/bestfaces/ folder (named person_XXXX.jpg) for quick
        visual review of everyone detected, without opening each
        person_XXXX/ subfolder individually.
        """
        import shutil

        gallery_dir = self.paths.dataset_dir / "bestfaces"
        gallery_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for person_dir in sorted(self.paths.dataset_dir.iterdir()):
            if not person_dir.is_dir() or person_dir.name in ("unknown", "bestfaces"):
                continue
            # Filename is now "best_face.jpg" (no timestamp available) or
            # "best_face_<YYYYMMDD_HHMMSS>.jpg" (timestamp available) —
            # there's only ever one such file per person.
            matches = sorted(person_dir.glob("best_face*.jpg"))
            if matches:
                shutil.copy2(matches[0], gallery_dir / f"{person_dir.name}.jpg")
                count += 1

        logger.info("Built bestfaces gallery: %d images in %s", count, gallery_dir)

    def _write_metadata_summary(self, summary: dict) -> None:
        meta_path = self.paths.metadata_json_file()
        existing = []
        if meta_path.exists():
            try:
                existing = json.loads(meta_path.read_text(encoding="utf-8"))
                if not isinstance(existing, list):
                    existing = [existing]
            except (json.JSONDecodeError, OSError):
                existing = []
        existing.append(summary)
        meta_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")