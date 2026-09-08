"""
utils/quality.py
=================

Computes quality metrics for a face crop (blur, brightness, resolution,
pose). These are used purely as METADATA TAGS stored in faces.db — per
`dataset.save_all_faces: true` in config.yaml, nothing here causes an
image to be skipped. This lets you later query faces.db to filter the
dataset for training ("only use sharp, front-facing images") without
having discarded the blurry/side-angle ones at collection time.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from utils.config_loader import Config


@dataclass
class QualityReport:
    is_blurry: bool
    laplacian_var: float
    brightness_ok: bool
    mean_brightness: float
    pose_ok: bool
    is_low_res: bool
    width: int
    height: int


def assess_quality(crop: np.ndarray, yaw: float, pitch: float, config: Config) -> QualityReport:
    """
    Compute quality tags for a single face crop. Never raises, never
    filters — always returns a report, even for a low-quality image.
    """
    qf = config.quality_filters
    h, w = crop.shape[:2]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_blurry = laplacian_var < qf.blur.laplacian_var_threshold if qf.blur.enabled else False

    mean_brightness = float(np.mean(gray))
    brightness_ok = (
        qf.brightness.min_mean_intensity <= mean_brightness <= qf.brightness.max_mean_intensity
        if qf.brightness.enabled else True
    )

    pose_ok = (
        abs(yaw) <= qf.pose.max_yaw_degrees and abs(pitch) <= qf.pose.max_pitch_degrees
        if qf.pose.enabled else True
    )

    is_low_res = w < qf.min_resolution.width or h < qf.min_resolution.height

    return QualityReport(
        is_blurry=is_blurry,
        laplacian_var=laplacian_var,
        brightness_ok=brightness_ok,
        mean_brightness=mean_brightness,
        pose_ok=pose_ok,
        is_low_res=is_low_res,
        width=w,
        height=h,
    )


def composite_quality_score(
    quality: QualityReport,
    detection_confidence: float,
    yaw: float,
    pitch: float,
) -> float:
    """
    Combines multiple signals into ONE score used to pick the single BEST
    face image per person (see dataset.best_face_only in config.yaml).

    Higher is better. Components, each normalized to roughly [0, 1] before
    weighting:

        sharpness   (40%) — Laplacian variance, saturating around 300
                             (typical "sharp" threshold for small face crops)
        brightness  (20%) — how close to ideal mid-range brightness (~130)
        pose        (25%) — how close to frontal (yaw=0, pitch=0)
        confidence  (15%) — the detector's own confidence score

    Weights favor sharpness and pose because those are what make a face
    crop USEFUL for recognition — a high-confidence detection of a
    blurry, extreme side-angle face is still a poor "best face".
    """
    sharpness_score = min(quality.laplacian_var / 300.0, 1.0)

    ideal_brightness = 130.0
    brightness_score = max(0.0, 1.0 - abs(quality.mean_brightness - ideal_brightness) / ideal_brightness)

    pose_penalty = (abs(yaw) + abs(pitch)) / 180.0  # 0 = perfectly frontal, 1 = worst case
    pose_score = max(0.0, 1.0 - pose_penalty)

    confidence_score = max(0.0, min(detection_confidence, 1.0))

    # Low-resolution crops are heavily penalized regardless of other scores
    # — a sharp-looking but tiny/upsampled crop is still not a good "best face".
    resolution_penalty = 0.5 if quality.is_low_res else 0.0

    score = (
        0.40 * sharpness_score
        + 0.20 * brightness_score
        + 0.25 * pose_score
        + 0.15 * confidence_score
    ) - resolution_penalty

    return float(score)
