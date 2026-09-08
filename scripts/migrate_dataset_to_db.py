"""scripts/migrate_dataset_to_db.py

Migrate existing files under dataset/person_*/ into database/faces.db
by inserting each image as a JPEG blob. Uses the project's
PathManager and FacesDatabase APIs so behaviour matches runtime.

Usage:
    python scripts/migrate_dataset_to_db.py
    python scripts/migrate_dataset_to_db.py --root /path/to/project
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

from utils.path_manager import PathManager
from utils.database import FacesDatabase


def migrate(root: str | None = None, db_path: str | None = None) -> None:
    paths = PathManager(root_dir=root or ".")
    if db_path:
        faces_db = FacesDatabase(Path(db_path))
    else:
        faces_db = FacesDatabase(paths.faces_db_file())

    dataset_dir = paths.dataset_dir
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return

    skipped = 0
    inserted = 0

    for person_dir in sorted(dataset_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        if person_dir.name in ("unknown", "bestfaces"):
            continue

        person_id = person_dir.name
        for img_path in sorted(person_dir.glob("**/*")):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            try:
                with open(img_path, "rb") as f:
                    blob = f.read()
            except OSError:
                print(f"Could not read {img_path}; skipping.")
                skipped += 1
                continue

            # Try to decode to get width/height
            arr = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                # skip unreadable images
                print(f"Could not decode {img_path}; skipping.")
                skipped += 1
                continue
            height, width = arr.shape[:2]

            faces_db.insert_face(
                person_id=person_id,
                track_id=0,
                image_path=None,
                image_blob=blob,
                source_video="[migrated]",
                frame_number=-1,
                confidence=0.0,
                embedding=None,
                quality_score=0.0,
                is_blurry=False,
                laplacian_var=0.0,
                brightness_ok=True,
                mean_brightness=0.0,
                pose_ok=True,
                yaw=0.0,
                pitch=0.0,
                is_low_res=False,
                width=width,
                height=height,
                is_best_face=False,
                landmarks=None,
            )
            inserted += 1
            print(f"Inserted {img_path} -> DB")

    print(f"Done. Inserted: {inserted}, skipped: {skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="Project root directory (default: current)")
    parser.add_argument("--db", help="Optional path to SQLite DB file to write to (overrides default)")
    args = parser.parse_args()
    migrate(args.root, args.db)
