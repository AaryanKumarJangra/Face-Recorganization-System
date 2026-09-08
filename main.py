"""
main.py
=======

Entry point for FaceRecognitionSystem.

Commands
--------
    python main.py --check-setup
        Verify config/paths/logging/device resolution.

    python main.py --build-dataset --video videos/sample.mp4
        Run detection + tracking on a video, save EVERY detected face
        crop into dataset/person_XXXX/ (see dataset.save_all_faces in
        configs/config.yaml), and record metadata in database/faces.db.

    python main.py --train
        Build the embeddings gallery from dataset/, train SVM + KNN
        classifiers, save them to models/recognition/.

    python main.py --recognize --video videos/new_video.mp4
        Run the trained classifier on a new video, annotate faces with
        name/confidence/track ID, save the output video, and log every
        recognition event to database/recognition_logs.db.

Bulk mode
---------
    python main.py --build-dataset --video-dir videos/
    python main.py --recognize --video-dir videos/

    Instead of --video <single file>, point --video-dir at a folder and
    EVERY video inside it is processed in one command, one after another,
    IN CHRONOLOGICAL ORDER based on the timestamp in each filename (e.g.
    CAMID_20260715_010447.mp4) — not just alphabetical — so the dataset
    grows the same way it would if you ran each video by hand in time
    order. Files with no recognizable timestamp are processed last.
    Supported formats: .mp4 .avi .mov .mkv .wmv .flv .ts .m4v .h264 .264
    (raw/elementary H.264 streams are opened directly via OpenCV/FFmpeg —
    no container required). The detector/classifier model is loaded only
    once for the whole batch; only the per-video tracker is reset between
    files, so identity numbering stays consistent across the batch and
    processing a folder of 50 videos is not 50x slower to start up.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

from utils.config_loader import Config
from utils.logger import get_logger
from utils.path_manager import PathManager

logger = get_logger(__name__, log_filename="main.log")

# Extensions considered when --video-dir is given. Includes raw H.264
# elementary streams (.h264 / .264) alongside common containers.
VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".ts", ".m4v",
    ".h264", ".264",
}

# Matches a "..._YYYYMMDD_HHMMSS..." timestamp inside a video filename, e.g.
# A05401C4F5FC_20260715_010447.mp4
_TIMESTAMP_RE = re.compile(r"(\d{8})_(\d{6})")


def _video_sort_key(path: Path):
    """
    Sort key that orders videos by the capture timestamp embedded in their
    filename (CAMID_YYYYMMDD_HHMMSS.ext), not plain alphabetical order.
    This keeps dataset growth (person numbering, best-face selection,
    cross-video re-ID) following real chronological order even when videos
    from different cameras/prefixes are mixed in the same folder. Files
    without a recognizable timestamp sort after timestamped ones, in
    alphabetical order among themselves.
    """
    match = _TIMESTAMP_RE.search(path.stem)
    if match:
        try:
            dt = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
            return (0, dt, path.name)
        except ValueError:
            pass
    return (1, datetime.max, path.name)


def collect_videos(video_dir: str) -> list[str]:
    """
    Return video file paths inside `video_dir`, ordered by the timestamp
    embedded in each filename (falls back to alphabetical if no timestamp
    is found), so bulk mode processes them in the same chronological order
    you'd get running them one-by-one yourself.
    """
    folder = Path(video_dir)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {video_dir}")
    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    files.sort(key=_video_sort_key)
    return [str(p) for p in files]


def check_setup(config_path: str = "configs/config.yaml") -> None:
    """Verify configuration, paths, and logging are working."""
    cfg = Config(config_path)
    paths = PathManager(root_dir=cfg.project.root_dir)

    print("\n✅ Environment setup verified successfully.")
    print(f"   Project root : {paths.root}")
    print(f"   Device       : {cfg.resolved_device}")
    print(f"   Log file     : {paths.log_file('setup.log')}")


def build_dataset(config_path: str, video_path: str) -> None:
    """Run detection + tracking on a video and save every face crop."""
    from training.dataset_builder import DatasetBuilder

    cfg = Config(config_path)
    paths = PathManager(root_dir=cfg.project.root_dir)

    builder = DatasetBuilder(cfg, paths)
    summary = builder.process_video(video_path)

    print("\n✅ Dataset build complete.")
    print(f"   Video processed      : {summary['source_video']}")
    print(f"   Frames processed     : {summary['total_frames_processed']}")
    print(f"   Face crops saved     : {summary['total_faces_saved']}")
    print(f"   Unique people found  : {summary['unique_people_detected']}")
    print(f"   Avg processing FPS   : {summary['average_fps']}")
    print(f"   Dataset location     : {paths.dataset_dir}")
    print(f"   Metadata DB          : {paths.faces_db_file()}")


def build_dataset_batch(config_path: str, video_paths: list[str]) -> None:
    """Run --build-dataset over every video in video_paths, in one command."""
    from training.dataset_builder import DatasetBuilder

    cfg = Config(config_path)
    paths = PathManager(root_dir=cfg.project.root_dir)

    builder = DatasetBuilder(cfg, paths)  # detector loaded once for the whole batch

    print(f"\n▶ Bulk dataset build starting on {len(video_paths)} video(s).")
    total_faces = 0
    failures = []
    for i, video_path in enumerate(video_paths, start=1):
        print(f"\n[{i}/{len(video_paths)}] {video_path}")
        try:
            summary = builder.process_video(video_path)
        except Exception as exc:  # keep going even if one file is bad/corrupt
            logger.exception("Failed to process %s", video_path)
            print(f"   ❌ Failed: {exc}")
            failures.append((video_path, str(exc)))
            continue
        total_faces += summary["total_faces_saved"]
        print(f"   ✅ frames={summary['total_frames_processed']} "
              f"faces_saved={summary['total_faces_saved']} "
              f"fps={summary['average_fps']}")

    print("\n✅ Bulk dataset build complete.")
    print(f"   Videos processed     : {len(video_paths) - len(failures)}/{len(video_paths)}")
    print(f"   Total face crops     : {total_faces}")
    print(f"   Unique people so far : {builder.tracker.total_unique_people() if builder.tracker else 0}")
    print(f"   Dataset location     : {paths.dataset_dir}")
    print(f"   Metadata DB          : {paths.faces_db_file()}")
    if failures:
        print(f"   ⚠️  {len(failures)} video(s) failed:")
        for video_path, err in failures:
            print(f"      - {video_path}: {err}")


def recognize_batch(config_path: str, video_paths: list[str]) -> None:
    """Run --recognize over every video in video_paths, in one command."""
    from recognition.recognize import FaceRecognizer

    cfg = Config(config_path)
    paths = PathManager(root_dir=cfg.project.root_dir)

    recognizer = FaceRecognizer(cfg, paths)  # detector+classifier loaded once

    print(f"\n▶ Bulk recognition starting on {len(video_paths)} video(s).")
    outputs = []
    failures = []
    for i, video_path in enumerate(video_paths, start=1):
        print(f"\n[{i}/{len(video_paths)}] {video_path}")
        try:
            output_path = recognizer.process_video(video_path)
        except Exception as exc:
            logger.exception("Failed to process %s", video_path)
            print(f"   ❌ Failed: {exc}")
            failures.append((video_path, str(exc)))
            continue
        outputs.append(output_path)
        print(f"   ✅ Output: {output_path}")

    print("\n✅ Bulk recognition complete.")
    print(f"   Videos processed : {len(outputs)}/{len(video_paths)}")
    for output_path in outputs:
        print(f"   - {output_path}")
    if failures:
        print(f"   ⚠️  {len(failures)} video(s) failed:")
        for video_path, err in failures:
            print(f"      - {video_path}: {err}")


def train(config_path: str) -> None:
    """Build embeddings gallery and train SVM + KNN classifiers."""
    from training.train_classifier import ClassifierTrainer

    cfg = Config(config_path)
    paths = PathManager(root_dir=cfg.project.root_dir)

    trainer = ClassifierTrainer(cfg, paths)
    results = trainer.train()

    print("\n✅ Training complete.")
    for method, result in results.items():
        print(f"   {method.upper():5s} accuracy: {result['accuracy']:.4f}  -> {result['model_path']}")


def recognize(config_path: str, video_path: str) -> None:
    """Run trained classifiers on a new video (Phase 7)."""
    from recognition.recognize import FaceRecognizer

    cfg = Config(config_path)
    paths = PathManager(root_dir=cfg.project.root_dir)

    recognizer = FaceRecognizer(cfg, paths)
    output_path = recognizer.process_video(video_path)

    print("\n✅ Recognition complete.")
    print(f"   Output video: {output_path}")


def consolidate(config_path: str) -> None:
    """Re-cluster all saved crops by embedding similarity to merge fragmented person folders."""
    from training.consolidate_identities import IdentityConsolidator

    cfg = Config(config_path)
    paths = PathManager(root_dir=cfg.project.root_dir)

    consolidator = IdentityConsolidator(cfg, paths)
    summary = consolidator.run()

    print("\n✅ Consolidation complete.")
    print(f"   Person folders before: {summary['before']}")
    print(f"   Person folders after : {summary['after']}")
    print(f"   Images moved          : {summary.get('images_moved', 0)}")


def augment(config_path: str, num_variants: int) -> None:
    """Expand each person's best face into many synthetic training embeddings."""
    from training.augment_dataset import DatasetAugmenter

    cfg = Config(config_path)
    paths = PathManager(root_dir=cfg.project.root_dir)

    augmenter = DatasetAugmenter(cfg, paths)
    summary = augmenter.run(num_variants=num_variants)

    print("\n✅ Augmentation complete.")
    print(f"   People augmented          : {summary['people_augmented']}")
    print(f"   Synthetic embeddings made : {summary['total_synthetic_embeddings']}")
    for pid, count in summary["per_person"].items():
        print(f"     {pid}: +{count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FaceRecognitionSystem")
    parser.add_argument("--check-setup", action="store_true", help="Verify environment setup.")
    parser.add_argument("--build-dataset", action="store_true", help="Detect, track, and save faces from a video.")
    parser.add_argument("--train", action="store_true", help="Train SVM + KNN classifiers on the dataset.")
    parser.add_argument("--consolidate", action="store_true", help="Merge fragmented person folders by re-clustering embeddings.")
    parser.add_argument("--augment", action="store_true", help="Generate synthetic training embeddings from each person's best face.")
    parser.add_argument("--num-variants", type=int, default=20, help="Augmented variants to generate per person (used with --augment).")
    parser.add_argument("--recognize", action="store_true", help="Recognize faces in a new video.")
    parser.add_argument("--video", type=str, default=None, help="Path to a single input video (for --build-dataset / --recognize).")
    parser.add_argument("--video-dir", type=str, default=None, help="Folder of videos to process in bulk with one command (for --build-dataset / --recognize). Supports .mp4/.avi/.mov/.mkv/.wmv/.flv/.ts/.m4v/.h264/.264.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to the YAML config file.")
    args = parser.parse_args()

    if args.check_setup:
        check_setup(args.config)
    elif args.build_dataset:
        if args.video_dir:
            videos = collect_videos(args.video_dir)
            if not videos:
                parser.error(f"No video files found in {args.video_dir}")
            build_dataset_batch(args.config, videos)
        elif args.video:
            build_dataset(args.config, args.video)
        else:
            parser.error("--build-dataset requires --video <path> or --video-dir <folder>")
    elif args.train:
        train(args.config)
    elif args.consolidate:
        consolidate(args.config)
    elif args.augment:
        augment(args.config, args.num_variants)
    elif args.recognize:
        if args.video_dir:
            videos = collect_videos(args.video_dir)
            if not videos:
                parser.error(f"No video files found in {args.video_dir}")
            recognize_batch(args.config, videos)
        elif args.video:
            recognize(args.config, args.video)
        else:
            parser.error("--recognize requires --video <path> or --video-dir <folder>")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()