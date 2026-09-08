"""
utils/augmentation.py
=======================

Generates augmented variants of a face crop to expand limited training
data per person — the practical alternative to fine-tuning on massive
public datasets (WiderFace/SCFace/SurvFace/VGGFace2/MS1MV3), which would
need GPU cluster time we don't have here.

From ~1 best_face.jpg (or however many real crops you have) per person,
this generates realistic variants matching common CCTV degradation:
rotation, brightness/contrast shifts, motion blur, Gaussian blur, JPEG
compression artifacts, Gaussian noise, and resolution downscale/upscale
(simulating a face captured at different distances from camera).

NOT included (deliberately): synthetic rain/fog overlays and random
occlusion patches. These help for general scene/object recognition
augmentation but rarely reflect how a face is actually degraded in real
CCTV footage, and risk teaching the classifier to key off inserted
artifacts rather than the actual face. Motion blur + compression + noise
already covers the realistic degradation modes for this use case.

IMPORTANT — why every function returns (image, landmarks) pairs:
To get a USABLE embedding from an augmented crop, we need to properly
re-align it using InsightFace's own 5-point landmark warp (we verified
empirically: naive resize-to-112x112 without alignment only gives ~0.54
cosine similarity to the ground-truth embedding, vs 0.996+ with proper
landmark alignment). Geometric augmentations (rotation, crop) move the
face within the image, so the landmarks must be transformed identically
or alignment — and therefore the resulting embedding — will be wrong.
"""

from __future__ import annotations

import io
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image


def rotate(img: np.ndarray, landmarks: np.ndarray, angle_degrees: float) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    rotated_img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

    ones = np.ones((landmarks.shape[0], 1))
    pts = np.hstack([landmarks, ones])
    rotated_landmarks = (matrix @ pts.T).T
    return rotated_img, rotated_landmarks


def adjust_brightness(img: np.ndarray, landmarks: np.ndarray, factor: float) -> Tuple[np.ndarray, np.ndarray]:
    """factor > 1 brightens, < 1 darkens. Landmarks unchanged (no geometry change)."""
    return cv2.convertScaleAbs(img, alpha=factor, beta=0), landmarks


def motion_blur(img: np.ndarray, landmarks: np.ndarray, kernel_size: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = np.ones(kernel_size)
    kernel /= kernel_size
    return cv2.filter2D(img, -1, kernel), landmarks


def gaussian_blur(img: np.ndarray, landmarks: np.ndarray, kernel_size: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0), landmarks


def jpeg_compress(img: np.ndarray, landmarks: np.ndarray, quality: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Simulates heavy JPEG compression artifacts typical of stored CCTV footage."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = np.array(Image.open(buffer).convert("RGB"))
    return cv2.cvtColor(compressed, cv2.COLOR_RGB2BGR), landmarks


def gaussian_noise(img: np.ndarray, landmarks: np.ndarray, sigma: float = 15.0) -> Tuple[np.ndarray, np.ndarray]:
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8), landmarks


def resolution_shift(img: np.ndarray, landmarks: np.ndarray, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """Downscale then upscale back to the SAME size — simulates a face
    captured farther from camera. Output dimensions unchanged, so
    landmarks stay valid."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR), landmarks


def random_crop_pad(img: np.ndarray, landmarks: np.ndarray, crop_fraction: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    """Slightly shifts framing — simulates imperfect bounding box crops."""
    h, w = img.shape[:2]
    ch, cw = int(h * crop_fraction), int(w * crop_fraction)
    y0 = np.random.randint(0, max(1, h - ch + 1))
    x0 = np.random.randint(0, max(1, w - cw + 1))
    cropped = img[y0:y0 + ch, x0:x0 + cw]
    resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    scale_x, scale_y = w / cw, h / ch
    new_landmarks = landmarks.copy()
    new_landmarks[:, 0] = (landmarks[:, 0] - x0) * scale_x
    new_landmarks[:, 1] = (landmarks[:, 1] - y0) * scale_y
    return resized, new_landmarks


def generate_augmented_set(
    img: np.ndarray,
    landmarks: np.ndarray,
    num_variants: int = 20,
    seed: int | None = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates `num_variants` augmented (image, landmarks) pairs — each a
    random combination of 1-3 augmentation operations, mirroring
    realistic combined degradation (e.g. blurry AND compressed AND
    darker at once).

    `landmarks` must be in `img`-relative pixel coordinates (i.e. the
    5-point landmarks adjusted for this specific crop's offset from the
    original frame — see training/augment_dataset.py for how these are
    computed from stored detection data).

    Returns the augmented (image, landmarks) pairs only — does NOT
    include the original.
    """
    rng = np.random.RandomState(seed)
    ops = [
        lambda x, lm: rotate(x, lm, rng.uniform(-20, 20)),
        lambda x, lm: adjust_brightness(x, lm, rng.uniform(0.6, 1.5)),
        lambda x, lm: motion_blur(x, lm, int(rng.choice([5, 7, 9]))),
        lambda x, lm: gaussian_blur(x, lm, int(rng.choice([3, 5, 7]))),
        lambda x, lm: jpeg_compress(x, lm, int(rng.randint(20, 60))),
        lambda x, lm: gaussian_noise(x, lm, rng.uniform(5, 25)),
        lambda x, lm: resolution_shift(x, lm, rng.uniform(0.3, 0.7)),
        lambda x, lm: random_crop_pad(x, lm, rng.uniform(0.8, 0.95)),
    ]

    variants = []
    for _ in range(num_variants):
        result_img, result_lm = img.copy(), landmarks.copy()
        n_ops = rng.randint(1, 4)  # combine 1-3 augmentations per variant
        chosen_ops = rng.choice(len(ops), size=n_ops, replace=False)
        for op_idx in chosen_ops:
            try:
                result_img, result_lm = ops[op_idx](result_img, result_lm)
            except Exception:
                continue  # skip a failed op, keep prior state
        variants.append((result_img, result_lm))

    return variants

