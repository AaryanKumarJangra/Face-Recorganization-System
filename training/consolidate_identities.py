"""
training/consolidate_identities.py
=====================================

Post-processing fix for tracker fragmentation.

Why this exists
-----------------
The live tracker (tracking/face_tracker.py) makes ID decisions frame-by-frame
with no ability to look ahead or compare against the WHOLE video at once.
On real-world footage this causes fragmentation: a person whose face size
changes quickly (e.g. walking toward the camera) can break IoU matching
faster than re-ID can catch up, especially on tiny/low-quality crops where
the embedding itself is less reliable. The result: one real person ends up
split across several person_XXXX folders.

This script fixes that AFTER the fact, with the advantage of seeing every
face in the whole video at once:

    1. Load every saved crop's embedding from database/faces.db (stored at
       COLLECTION time — see utils/database.py). We deliberately do NOT
       re-run face detection on the saved crop images: RetinaFace relies
       on scene-level context that a tight, already-cropped face image
       doesn't have, which makes re-detection on crops unreliable (we
       measured >90% failure re-detecting on real saved crops).
    2. Cluster embeddings by cosine similarity (agglomerative clustering —
       no need to pre-specify the number of people).
    3. Merge folders whose crops cluster together into one canonical
       person_XXXX folder, physically moving files.
    4. Update database/faces.db so person_id columns match the merge.
    5. Print a before/after summary.

This does NOT delete or reject any image — it only relabels/moves them
under the correct (merged) person folder, consistent with
dataset.save_all_faces: true.

Usage
-----
    python main.py --consolidate
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from utils.config_loader import Config
from utils.database import FacesDatabase
from utils.logger import get_logger
from utils.path_manager import PathManager

logger = get_logger(__name__, log_filename="consolidate.log")


class IdentityConsolidator:
    def __init__(self, config: Config, paths: PathManager) -> None:
        self.config = config
        self.paths = paths
        self.faces_db = FacesDatabase(paths.faces_db_file())

    def run(self) -> dict:
        if not self.config.dataset.best_face_only:
            raise RuntimeError(
                "Consolidation currently only supports dataset.best_face_only: true "
                "in configs/config.yaml (it assumes exactly one physical image per "
                "person). Set best_face_only to true and re-run --build-dataset, "
                "or skip --consolidate if you're intentionally keeping every frame."
            )

        rows = self.faces_db.get_all_embeddings_full()
        if len(rows) < 2:
            logger.warning("Fewer than 2 observations with embeddings found — nothing to consolidate.")
            return {"before": 0, "after": 0, "images_moved": 0}

        old_person_ids = [r["person_id"] for r in rows]
        before_count = len(set(old_person_ids))
        embeddings = np.array(
            [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows], dtype=np.float32
        )

        logger.info("Loaded %d observations across %d folders from faces.db for consolidation.",
                     len(rows), before_count)

        cluster_labels = self._cluster(embeddings)
        merge_summary = self._apply_merge(rows, cluster_labels)
        merge_summary["before"] = before_count
        return merge_summary

    def _cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Agglomerative clustering on cosine distance. distance_threshold is
        derived from tracking.reid.similarity_threshold (cosine similarity)
        converted to cosine distance (1 - similarity), so consolidation
        uses the same notion of "same person" as the live tracker did.
        """
        similarity_threshold = self.config.tracking.reid.similarity_threshold
        distance_threshold = 1.0 - similarity_threshold

        # Normalize embeddings so euclidean distance on unit vectors
        # approximates cosine distance well enough for clustering.
        norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="single",   # single linkage merges chains of gradually-varying
                                 # poses/lighting within one person's appearance far
                                 # better than average linkage, which is thrown off by
                                 # the high intra-person variance typical of real footage
        )
        labels = clustering.fit_predict(norm)
        logger.info("Clustering found %d distinct identities from %d images.",
                     len(set(labels)), len(labels))
        return labels

    def _apply_merge(self, rows, cluster_labels: np.ndarray) -> dict:
        # Group row indices by cluster.
        clusters: Dict[int, List[int]] = {}
        for idx, cluster in enumerate(cluster_labels):
            clusters.setdefault(int(cluster), []).append(idx)

        images_moved = 0
        final_person_ids = set()

        for cluster_indices in clusters.values():
            cluster_rows = [rows[i] for i in cluster_indices]
            old_ids_in_cluster = sorted({r["person_id"] for r in cluster_rows})

            # Canonical person_id: the lowest-numbered original person_id
            # in this cluster — keeps IDs stable/predictable after merges.
            canonical_id = old_ids_in_cluster[0]
            final_person_ids.add(canonical_id)

            # Pick the single best row in the WHOLE cluster (across all
            # merged former person_ids) by quality_score, to become the
            # new best_face.jpg for the canonical person.
            scored_rows = [r for r in cluster_rows if r["quality_score"] is not None]
            best_row = max(scored_rows, key=lambda r: r["quality_score"]) if scored_rows else cluster_rows[0]

            canonical_dir = self.paths.get_person_dir(canonical_id)
            canonical_best_path = canonical_dir / "best_face.jpg"

            # Physically consolidate: only move a file if the best row's
            # current image isn't already the canonical file.
            best_src = Path(best_row["image_path"])
            if best_src.resolve() != canonical_best_path.resolve():
                if best_src.exists():
                    shutil.move(str(best_src), str(canonical_best_path))
                    images_moved += 1

            # Remove any OTHER physical best_face.jpg files belonging to
            # non-canonical old person_ids in this cluster (each old
            # person_id has at most one physical file in best_face_only
            # mode) — we already kept the winner above.
            for old_id in old_ids_in_cluster:
                if old_id == canonical_id:
                    continue
                stale_path = self.paths.dataset_dir / old_id / "best_face.jpg"
                if stale_path.exists() and stale_path.resolve() != canonical_best_path.resolve():
                    stale_path.unlink()

            # Update DB: every row in this cluster now belongs to
            # canonical_id and points at the one consolidated file; only
            # best_row is flagged is_best_face=1.
            row_ids = [r["id"] for r in cluster_rows]
            self.faces_db.reassign_rows(
                row_ids=row_ids,
                new_person_id=canonical_id,
                new_image_path=str(canonical_best_path),
                best_row_id=best_row["id"],
            )

        self._remove_empty_person_dirs()
        self._rebuild_bestfaces_gallery()

        after_count = len(final_person_ids)
        summary = {"after": after_count, "images_moved": images_moved}
        logger.info("Consolidation complete: after=%d images_moved=%d", after_count, images_moved)
        return summary

    def _rebuild_bestfaces_gallery(self) -> None:
        gallery_dir = self.paths.dataset_dir / "bestfaces"
        if gallery_dir.exists():
            shutil.rmtree(gallery_dir)
        gallery_dir.mkdir(parents=True, exist_ok=True)

        for person_dir in sorted(self.paths.dataset_dir.iterdir()):
            if not person_dir.is_dir() or person_dir.name in ("unknown", "bestfaces"):
                continue
            best_face = person_dir / "best_face.jpg"
            if best_face.exists():
                shutil.copy2(best_face, gallery_dir / f"{person_dir.name}.jpg")

    def _remove_empty_person_dirs(self) -> None:
        for d in self.paths.dataset_dir.iterdir():
            if d.is_dir() and d.name != "unknown" and not any(d.iterdir()):
                d.rmdir()
