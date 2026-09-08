"""
recognition/recognize.py
==========================

Phase 7: recognize faces in a new video using the classifier(s) trained
by training/train_classifier.py.

For each detected+tracked face:
    - Embed it (InsightFace ArcFace, same model as training).
    - Predict identity with the SVM classifier (probability score).
    - If probability < recognition.similarity_threshold -> label "Unknown".
    - Draw bounding box, name, confidence, track ID, and FPS on the frame.
    - Log the event to database/recognition_logs.db.
    - Write the annotated video to output/.

Usage
-----
    python main.py --recognize --video videos/new_video.mp4
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import cv2

from detection.face_detector import FaceDetector
from tracking.face_tracker import FaceTracker
from utils.config_loader import Config
from utils.database import RecognitionLogsDatabase
from utils.logger import get_logger
from utils.path_manager import PathManager

logger = get_logger(__name__, log_filename="recognition.log")


class FaceRecognizer:
    """Runs the trained SVM classifier on a video and writes an annotated output."""

    def __init__(self, config: Config, paths: PathManager) -> None:
        self.config = config
        self.paths = paths

        self.detector = FaceDetector(config, paths)
        # Tracker is (re)created per-video inside process_video() so one
        # FaceRecognizer instance can safely process a whole batch of
        # videos without reloading the detector/classifier each time.
        self.tracker = None
        self.logs_db = RecognitionLogsDatabase(paths.recognition_logs_db_file())

        self.classifier = self._load_classifier()

    def _load_classifier(self):
        model_path = self.paths.recognition_model_file("svm_classifier.pkl")
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained classifier found at {model_path}. "
                f"Run 'python main.py --train' first."
            )
        with open(model_path, "rb") as f:
            clf = pickle.load(f)
        logger.info("Loaded classifier from %s", model_path)
        return clf

    def process_video(self, video_path: str) -> str:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        # Fresh tracker per video — track IDs/buffers must not leak between
        # unrelated videos in a batch run.
        self.tracker = FaceTracker(self.config)

        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = self.paths.output_file(f"recognized_{Path(video_path).stem}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps_in, (width, height))

        disp_cfg = self.config.recognition.display
        threshold = self.config.recognition.similarity_threshold
        unknown_label = self.config.recognition.unknown_label

        frame_number = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = self.detector.detect(frame)
            tracked_faces = self.tracker.update(faces)

            elapsed = time.time() - start_time
            current_fps = frame_number / elapsed if elapsed > 0 else 0.0

            for tf in tracked_faces:
                df = tf.detected_face
                name, confidence, is_unknown = self._predict(df.embedding, threshold, unknown_label)

                self.logs_db.log_event(
                    video_source=str(video_path),
                    frame_number=frame_number,
                    track_id=tf.track_id,
                    predicted_person_id=name,
                    confidence=confidence,
                    is_unknown=is_unknown,
                )

                self._draw_annotation(frame, df.bbox, name, confidence, tf.person_id, current_fps, disp_cfg)

            writer.write(frame)
            frame_number += 1

        cap.release()
        writer.release()
        logger.info("Recognition complete. Output written to %s", output_path)
        return str(output_path)

    def _predict(self, embedding, threshold: float, unknown_label: str):
        probs = self.classifier.predict_proba([embedding])[0]
        best_idx = probs.argmax()
        confidence = float(probs[best_idx])
        predicted_label = self.classifier.classes_[best_idx]

        if confidence < threshold:
            return unknown_label, confidence, True
        return predicted_label, confidence, False

    @staticmethod
    def _draw_annotation(frame, bbox, name, confidence, person_id, fps, disp_cfg) -> None:
        x1, y1, x2, y2 = bbox
        color = (0, 200, 0) if name != "Unknown" else (0, 0, 220)

        if disp_cfg.show_bbox:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label_parts = []
        if disp_cfg.show_name:
            label_parts.append(name)
        if disp_cfg.show_confidence:
            label_parts.append(f"{confidence:.2f}")
        if disp_cfg.show_track_id:
            # Display the STABLE person ID (e.g. "person_0007" -> "ID:7"),
            # not the internal track_id — track_id intentionally changes
            # every time the tracker re-links a person (e.g. after
            # occlusion or a large position jump), even when it correctly
            # identifies them as the same individual. person_id is what
            # stays fixed for a given person across the whole video.
            try:
                short_id = int(person_id.split("_")[1])
            except (IndexError, ValueError):
                short_id = person_id
            label_parts.append(f"ID:{short_id}")
        label = " | ".join(label_parts)

        if label:
            cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if disp_cfg.show_fps:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)