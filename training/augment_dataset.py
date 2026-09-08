"""
training/augment_dataset.py
==============================

Expands each person's single best_face.jpg into many synthetic embeddings
for training — the practical alternative to fine-tuning on massive public
face datasets (which would need GPU cluster time and days of training we
don't have here).

Pipeline
--------
    1. Load each person's best_face.jpg + its stored crop-relative
       landmarks from faces.db (is_best_face = 1 rows).
    2. Generate N augmented (image, landmarks) variants — rotation,
       brightness, motion/Gaussian blur, JPEG compression, noise,
       resolution shift, crop jitter (see utils/augmentation.py).
    3. Re-align each variant using ITS transformed landmarks and extract
       an embedding directly via the ArcFace recognition submodel —
       NOT by re-running full face detection (which fails >90% of the
       time on isolated crops — see detection/face_detector.py docstring
       for the measurement that led to this design).
    4. Insert each as a new synthetic observation row in faces.db
       (person_id = same, image_path = the original best_face.jpg, no
       new image files are created) so training/train_classifier.py
       picks them up automatically alongside the real observations.

This does NOT touch any real data — it's purely additive rows in
faces.db. Safe to re-run: use --reset first (see main.py) to clear
previously-generated synthetic rows before regenerating.

Usage
-----
    python main.py --augment                    # 20 variants per person (default)
    python main.py --augment --num-variants 50   # more variants
"""

from __future__ import annotations

import cv2
import numpy as np

from detection.face_detector import FaceDetector
from utils.augmentation import generate_augmented_set
from utils.config_loader import Config
from utils.database import FacesDatabase
from utils.logger import get_logger
from utils.path_manager import PathManager

logger = get_logger(__name__, log_filename="augment.log")

SYNTHETIC_MARKER_TRACK_ID = -1  # distinguishes augmented rows from real observations


class DatasetAugmenter:
    def __init__(self, config: Config, paths: PathManager) -> None:
        self.config = config
        self.paths = paths
        self.detector = FaceDetector(config, paths)  # needed for embed_aligned_crop
        self.faces_db = FacesDatabase(paths.faces_db_file())

    def run(self, num_variants: int = 20) -> dict:
        best_face_rows = self.faces_db.get_best_faces()
        if not best_face_rows:
            raise RuntimeError(
                "No best-face rows found in faces.db. Run "
                "'python main.py --build-dataset --video <path>' first."
            )

        # Clear any previously-generated synthetic rows so re-running
        # --augment doesn't keep piling up duplicates.
        self._clear_previous_synthetic_rows()

        total_generated = 0
        per_person_counts = {}

        for row in best_face_rows:
            person_id = row["person_id"]
            image_path = row["image_path"]
            landmarks_blob = row["landmarks"]

            if landmarks_blob is None:
                logger.warning(
                    "No stored landmarks for %s (%s) — skipping augmentation for "
                    "this person. This can happen for data collected before "
                    "landmark storage was added; re-run --build-dataset to fix.",
                    person_id, image_path,
                )
                continue

            crop = None
            if image_path and str(image_path).startswith("db://"):
                # Load image bytes from DB
                row_id = int(str(image_path).split("db://", 1)[1])
                blob = self.faces_db.get_image_blob(row_id)
                if blob is None:
                    logger.warning("No image blob for %s (%s) — skipping.", person_id, image_path)
                    continue
                arr = np.frombuffer(blob, dtype=np.uint8)
                crop = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if crop is None:
                    logger.warning("Could not decode image blob for %s — skipping.", person_id)
                    continue
            else:
                crop = cv2.imread(image_path)
                if crop is None:
                    logger.warning("Could not read %s for %s — skipping.", image_path, person_id)
                    continue

            landmarks = np.frombuffer(landmarks_blob, dtype=np.float32).reshape(5, 2)

            variants = generate_augmented_set(crop, landmarks, num_variants=num_variants)
            count = 0
            for aug_img, aug_landmarks in variants:
                try:
                    embedding = self.detector.embed_aligned_crop(aug_img, aug_landmarks)
                except Exception as exc:
                    logger.debug("Skipping one variant for %s (alignment failed): %s", person_id, exc)
                    continue

                self.faces_db.insert_face(
                    person_id=person_id,
                    track_id=SYNTHETIC_MARKER_TRACK_ID,
                    image_path=image_path,   # no new file — points at the real best_face.jpg
                    source_video="[augmented]",
                    frame_number=-1,
                    confidence=row["quality_score"] or 0.5,
                    embedding=embedding,
                    quality_score=row["quality_score"] or 0.5,
                    is_blurry=False, laplacian_var=0.0, brightness_ok=True, mean_brightness=0.0,
                    pose_ok=True, yaw=0.0, pitch=0.0, is_low_res=False, width=0, height=0,
                    is_best_face=False,
                    landmarks=aug_landmarks,
                )
                count += 1

            per_person_counts[person_id] = count
            total_generated += count
            logger.info("Generated %d augmented embeddings for %s", count, person_id)

        summary = {
            "people_augmented": len(per_person_counts),
            "total_synthetic_embeddings": total_generated,
            "per_person": per_person_counts,
        }
        logger.info("Augmentation complete: %s", summary)
        return summary

    def _clear_previous_synthetic_rows(self) -> None:
        import sqlite3
        with sqlite3.connect(str(self.paths.faces_db_file())) as conn:
            deleted = conn.execute(
                "DELETE FROM faces WHERE track_id = ?", (SYNTHETIC_MARKER_TRACK_ID,)
            ).rowcount
            conn.commit()
        if deleted:
            logger.info("Cleared %d previously-generated synthetic rows before regenerating.", deleted)
