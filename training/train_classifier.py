"""
training/train_classifier.py
==============================

Phase 6: trains a face-recognition classifier on top of the dataset built
by dataset_builder.py.

Pipeline:
    1. Walk dataset/person_*/ directories.
    2. Re-run each saved crop through InsightFace to get its embedding
       (embeddings aren't stored per-image at collection time — only
       computed here, once, and cached).
    3. Save the full embeddings gallery to:
         - database/embeddings.npy      (raw array, for fast reload)
         - models/recognition/face_embeddings.pkl  (array + labels + paths)
    4. Train SVM and KNN classifiers on the embeddings, compare accuracy
       on a held-out split, and save the trained models to
       models/recognition/svm_classifier.pkl (and knn_classifier.pkl).

Usage
-----
    python main.py --train
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from detection.face_detector import FaceDetector
from utils.config_loader import Config
from utils.logger import get_logger
from utils.path_manager import PathManager

logger = get_logger(__name__, log_filename="training.log")


class ClassifierTrainer:
    """
    Builds the embeddings gallery from dataset/ and trains SVM + KNN
    classifiers for Phase 7 recognition.
    """

    def __init__(self, config: Config, paths: PathManager) -> None:
        self.config = config
        self.paths = paths
        # NOTE: no FaceDetector here — embeddings are read from faces.db
        # (stored at collection time), so training no longer needs to
        # load/run the detection model at all.

    def build_embeddings_gallery(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Load embeddings directly from database/faces.db (stored at
        collection time in dataset_builder.py), rather than re-running
        face detection on saved crop images — re-detecting on an
        already-cropped face is unreliable since RetinaFace relies on
        scene-level context a tight crop doesn't have.

        Excludes dataset/unknown/ crops implicitly, since those were never
        inserted under a confirmed person_id.

        Returns
        -------
        (embeddings, labels, image_paths)
        """
        from utils.database import FacesDatabase

        faces_db = FacesDatabase(self.paths.faces_db_file())
        image_paths, labels, embeddings = faces_db.get_all_embeddings()

        if len(embeddings) == 0:
            raise RuntimeError(
                f"No embeddings found in {self.paths.faces_db_file()}. "
                f"Run 'python main.py --build-dataset --video <path>' first."
            )

        logger.info("Loaded %d embeddings across %d people from faces.db.",
                     len(embeddings), len(set(labels)))

        self._save_gallery(embeddings, labels, image_paths)
        return embeddings, labels, image_paths

    def _save_gallery(self, embeddings: np.ndarray, labels: List[str], image_paths: List[str]) -> None:
        np.save(self.paths.embeddings_npy_file(), embeddings)

        pkl_path = self.paths.recognition_model_file("face_embeddings.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({"embeddings": embeddings, "labels": labels, "image_paths": image_paths}, f)

        logger.info("Saved embeddings gallery to %s and %s", self.paths.embeddings_npy_file(), pkl_path)

    def train(self) -> dict:
        """
        Full training entrypoint: build gallery, train SVM + KNN, compare,
        save both models. Returns a comparison summary dict.
        """
        embeddings, labels, _ = self.build_embeddings_gallery()

        unique_people = sorted(set(labels))
        if len(unique_people) < 2:
            raise RuntimeError(
                f"Need at least 2 distinct people to train a classifier; found "
                f"{len(unique_people)}. Add more people to dataset/ first."
            )

        # A stratified train/test split needs at least 2 samples per class.
        # People with only 1 saved image can't be evaluated this way yet —
        # exclude them from THIS training run (their images stay in
        # dataset/ untouched) rather than crashing the whole pipeline.
        from collections import Counter
        counts = Counter(labels)
        too_few = {pid for pid, c in counts.items() if c < 2}
        if too_few:
            logger.warning(
                "Excluding %d people with fewer than 2 images from training "
                "(need more footage of them first): %s",
                len(too_few), sorted(too_few),
            )
            keep_mask = [lbl not in too_few for lbl in labels]
            embeddings = embeddings[keep_mask]
            labels = [lbl for lbl, keep in zip(labels, keep_mask) if keep]
            unique_people = sorted(set(labels))

        if len(unique_people) < 2:
            raise RuntimeError(
                f"After excluding people with <2 images, only {len(unique_people)} "
                f"remain. Need at least 2 people with 2+ images each to train."
            )

        t_cfg = self.config.training
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels,
            test_size=t_cfg.test_split,
            random_state=t_cfg.random_seed,
            stratify=labels,
        )

        results = {}

        if "svm" in t_cfg.methods:
            results["svm"] = self._train_svm(X_train, y_train, X_test, y_test)

        if "knn" in t_cfg.methods:
            results["knn"] = self._train_knn(X_train, y_train, X_test, y_test)

        logger.info("Training comparison: %s", results)
        return results

    def _train_svm(self, X_train, y_train, X_test, y_test) -> dict:
        svm_cfg = self.config.training.svm
        logger.info("Training SVM classifier (kernel=%s)...", svm_cfg.kernel)

        start = time.time()
        clf = SVC(kernel=svm_cfg.kernel, probability=svm_cfg.probability, C=svm_cfg.C)
        clf.fit(X_train, y_train)
        train_time = time.time() - start

        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        logger.info("SVM accuracy: %.4f (trained in %.2fs)", acc, train_time)
        logger.info("SVM classification report:\n%s", classification_report(y_test, preds, zero_division=0))

        model_path = self.paths.recognition_model_file("svm_classifier.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)
        logger.info("Saved SVM model to %s", model_path)

        return {"accuracy": acc, "train_time_seconds": round(train_time, 2), "model_path": str(model_path)}

    def _train_knn(self, X_train, y_train, X_test, y_test) -> dict:
        knn_cfg = self.config.training.knn
        logger.info("Training KNN classifier (k=%d, metric=%s)...", knn_cfg.n_neighbors, knn_cfg.metric)

        start = time.time()
        clf = KNeighborsClassifier(n_neighbors=knn_cfg.n_neighbors, metric=knn_cfg.metric)
        clf.fit(X_train, y_train)
        train_time = time.time() - start

        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        logger.info("KNN accuracy: %.4f (trained in %.2fs)", acc, train_time)

        model_path = self.paths.recognition_model_file("knn_classifier.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)
        logger.info("Saved KNN model to %s", model_path)

        return {"accuracy": acc, "train_time_seconds": round(train_time, 2), "model_path": str(model_path)}
