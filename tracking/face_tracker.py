"""
tracking/face_tracker.py
==========================

Assigns a persistent ID to every face across a video.

Two layers, working together:

1. SHORT-TERM (frame-to-frame): a ByteTrack-style IoU tracker. Every frame,
   new detections are matched to existing tracks by bounding-box overlap.
   A track survives up to `tracking.track_buffer` frames with no matching
   detection (handles brief occlusion, motion blur, quick turns).

2. LONG-TERM (re-identification): when a detection can't be matched to any
   active track (i.e. it looks like a "new" person), its face embedding is
   compared against a gallery of embeddings from all PAST tracks in this
   video — including ones that expired long ago. If a strong match is
   found (cosine similarity above `tracking.reid.similarity_threshold`),
   the new track re-uses that person's original ID instead of minting a
   new one.

This two-layer design is what satisfies "keep tracking the same person even
if they leave and come back later" — plain ByteTrack alone only handles
short gaps bounded by track_buffer; it does NOT solve long absences.

Usage
-----
    tracker = FaceTracker(config)
    for frame in video:
        faces = detector.detect(frame)
        tracks = tracker.update(faces)   # list[TrackedFace], one per input face
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from detection.face_detector import DetectedFace
from utils.config_loader import Config
from utils.logger import get_logger

logger = get_logger(__name__, log_filename="tracking.log")


def _iou(box_a: tuple, box_b: tuple) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)

    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter_area / float(area_a + area_b - inter_area)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_norm, b_norm))


@dataclass
class TrackedFace:
    """A DetectedFace tagged with a persistent tracking ID and person label."""

    detected_face: DetectedFace
    track_id: int
    person_id: str          # e.g. "person_0001" — assigned once, reused via re-ID
    frames_since_seen: int = 0
    is_new_person: bool = False   # True on the frame this person_id was first minted


class _Track:
    """Internal bookkeeping for one active (or recently-lost) track."""

    def __init__(self, track_id: int, person_id: str, face: DetectedFace, max_gallery: int):
        self.track_id = track_id
        self.person_id = person_id
        self.bbox = face.bbox
        self.frames_since_seen = 0
        self.max_gallery = max_gallery
        self.embedding_gallery: List[np.ndarray] = [face.embedding]

    def update(self, face: DetectedFace) -> None:
        self.bbox = face.bbox
        self.frames_since_seen = 0
        self.embedding_gallery.append(face.embedding)
        if len(self.embedding_gallery) > self.max_gallery:
            self.embedding_gallery.pop(0)

    def mean_embedding(self) -> np.ndarray:
        return np.mean(self.embedding_gallery, axis=0)


class FaceTracker:
    """
    Stateful tracker — create ONE instance per video and call `.update()`
    once per frame with that frame's detections.

    Parameters
    ----------
    config : Config
        Reads the `tracking` section (track_buffer, match_thresh, reid.*).
    """

    def __init__(self, config: Config, start_person_num: int = 1) -> None:
        self.config = config
        t_cfg = config.tracking

        self.track_buffer: int = t_cfg.track_buffer
        self.match_thresh: float = t_cfg.match_thresh
        self.reid_enabled: bool = t_cfg.reid.enabled
        self.reid_similarity_threshold: float = t_cfg.reid.similarity_threshold
        self.gallery_size: int = t_cfg.reid.gallery_embeddings_per_id

        self._active_tracks: Dict[int, _Track] = {}
        self._lost_tracks: Dict[int, _Track] = {}   # kept forever for re-ID, this video only
        self._next_track_id: int = 1
        # start_person_num lets callers continue numbering across multiple
        # dataset-build runs (e.g. person_0007 onward) instead of always
        # restarting at person_0001 and accidentally merging different
        # people from different videos into the same folder. Note: this
        # does NOT re-identify people ACROSS videos/runs (the re-ID gallery
        # is per-run/per-video only) — it only avoids ID collisions.
        self._next_person_num: int = start_person_num

        logger.info(
            "FaceTracker initialized (track_buffer=%d, match_thresh=%.2f, reid_enabled=%s)",
            self.track_buffer, self.match_thresh, self.reid_enabled,
        )

    def _mint_person_id(self) -> str:
        person_id = f"person_{self._next_person_num:04d}"
        self._next_person_num += 1
        return person_id

    def _try_reid(self, face: DetectedFace, exclude_track_ids: set) -> Optional[Tuple[int, _Track]]:
        """
        Compare `face`'s embedding against BOTH:
          - expired/lost tracks (person absent longer than track_buffer), and
          - active tracks that simply weren't IoU-matched this frame (e.g.
            the person moved/jumped too far for bbox overlap to catch, even
            though they never technically left).
        Excludes any track_id already matched this frame.

        Returns (track_id, track) for the best match above
        `reid_similarity_threshold`, else None.
        """
        if not self.reid_enabled:
            return None

        candidates = dict(self._lost_tracks)
        for tid, track in self._active_tracks.items():
            if tid not in exclude_track_ids:
                candidates[tid] = track

        if not candidates:
            return None

        best_id: Optional[int] = None
        best_track: Optional[_Track] = None
        best_score = self.reid_similarity_threshold

        for tid, track in candidates.items():
            score = _cosine_similarity(face.embedding, track.mean_embedding())
            if score > best_score:
                best_score = score
                best_track = track
                best_id = tid

        if best_track is not None:
            logger.info(
                "Re-identified %s (similarity=%.3f, was %s).",
                best_track.person_id, best_score,
                "lost" if best_id in self._lost_tracks else "active-but-unmatched",
            )
            return best_id, best_track
        return None

    def update(self, faces: List[DetectedFace]) -> List[TrackedFace]:
        """
        Process one frame's worth of detections. Must be called once per
        frame, in video order.

        Returns
        -------
        list[TrackedFace]
            Same length/order as `faces`, each tagged with a track_id and
            person_id.
        """
        results: List[TrackedFace] = []
        unmatched_faces = list(enumerate(faces))
        matched_track_ids = set()

        # --- Step 1: IoU-match against currently active tracks -----------
        for track_id, track in self._active_tracks.items():
            best_idx, best_iou = None, 0.0
            for idx, face in unmatched_faces:
                if idx in matched_track_ids:
                    continue
                iou = _iou(track.bbox, face.bbox)
                if iou > best_iou:
                    best_iou, best_idx = iou, idx

            if best_idx is not None and best_iou >= self.match_thresh:
                face = faces[best_idx]
                track.update(face)
                results.append(
                    TrackedFace(
                        detected_face=face,
                        track_id=track_id,
                        person_id=track.person_id,
                    )
                )
                matched_track_ids.add(best_idx)

        # --- Step 2: remaining detections -> try re-ID, else new track ---
        already_matched_this_frame = {r.track_id for r in results}
        for idx, face in unmatched_faces:
            if idx in matched_track_ids:
                continue

            reid_result = self._try_reid(face, exclude_track_ids=already_matched_this_frame)
            is_new_person = False

            if reid_result is not None:
                old_track_id, reid_track = reid_result
                # Re-link under a new track_id but the SAME person_id — this
                # is what makes leave/re-enter (and same-frame position
                # jumps) resolve to one consistent identity.
                new_track_id = self._next_track_id
                self._next_track_id += 1
                revived = _Track(new_track_id, reid_track.person_id, face, self.gallery_size)
                revived.embedding_gallery = reid_track.embedding_gallery + [face.embedding]
                self._active_tracks[new_track_id] = revived
                self._active_tracks.pop(old_track_id, None)
                self._lost_tracks.pop(old_track_id, None)
                already_matched_this_frame.add(new_track_id)
                person_id = reid_track.person_id
                track_id = new_track_id
            else:
                person_id = self._mint_person_id()
                track_id = self._next_track_id
                self._next_track_id += 1
                self._active_tracks[track_id] = _Track(track_id, person_id, face, self.gallery_size)
                is_new_person = True
                logger.info("New person detected: %s (track_id=%d)", person_id, track_id)

            results.append(
                TrackedFace(
                    detected_face=face,
                    track_id=track_id,
                    person_id=person_id,
                    is_new_person=is_new_person,
                )
            )

        # --- Step 3: age out tracks that weren't matched this frame -------
        matched_ids_this_frame = {r.track_id for r in results}
        expired_ids = []
        for track_id, track in self._active_tracks.items():
            if track_id in matched_ids_this_frame:
                continue
            track.frames_since_seen += 1
            if track.frames_since_seen > self.track_buffer:
                expired_ids.append(track_id)

        for track_id in expired_ids:
            lost = self._active_tracks.pop(track_id)
            self._lost_tracks[track_id] = lost
            logger.debug(
                "Track %d (%s) lost after %d frames — kept in re-ID gallery.",
                track_id, lost.person_id, self.track_buffer,
            )

        return results

    def active_track_count(self) -> int:
        return len(self._active_tracks)

    def total_unique_people(self) -> int:
        return self._next_person_num - 1
