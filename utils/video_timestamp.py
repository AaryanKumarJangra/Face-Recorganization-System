# """
# utils/video_timestamp.py
# ==========================

# Many CCTV/DVR exports encode the camera ID and recording START time
# directly in the filename, e.g.:

#     A05401C4F5FC_20260715_010447.mp4
#     ^camera_id   ^date     ^time (HHMMSS)

# This module parses that out and computes the real wall-clock timestamp
# of any given frame (start time + frame_number / fps), so saved face
# images can be named after when they actually happened instead of a
# generic "best_face.jpg" or frame-counter filename.
# """

# from __future__ import annotations

# import re
# from dataclasses import dataclass
# from datetime import datetime, timedelta
# from pathlib import Path
# from typing import Optional

# # Matches an 8-digit date (YYYYMMDD) followed by an underscore and a
# # 6-digit time (HHMMSS) anywhere in the filename stem.
# _TIMESTAMP_RE = re.compile(r"(\d{8})_(\d{6})")


# @dataclass
# class VideoTimestampInfo:
#     camera_id: str            # whatever prefix precedes the timestamp, e.g. "A05401C4F5FC"
#     start_datetime: datetime  # recording start time parsed from the filename

#     def frame_datetime(self, frame_number: int, fps: float) -> datetime:
#         """Wall-clock time of a given frame, assuming constant fps from start_datetime."""
#         if fps <= 0:
#             fps = 25.0  # defensive fallback; shouldn't normally happen
#         return self.start_datetime + timedelta(seconds=frame_number / fps)

#     def frame_timestamp_str(self, frame_number: int, fps: float) -> str:
#         """e.g. '20260726_104201' -- HHMMSS, computed as start_time +
#         frame_number/fps using real datetime arithmetic, so seconds
#         correctly roll over past 59 into the next minute/hour (e.g.
#         104159 + 2s -> 104201, never 104161)."""
#         return self.frame_datetime(frame_number, fps).strftime("%Y%m%d_%H%M%S")


# def parse_video_timestamp(video_path: str) -> Optional[VideoTimestampInfo]:
#     """
#     Parse camera ID + recording start time out of a video filename.
#     Returns None if the filename doesn't contain a recognizable
#     <...>_<YYYYMMDD>_<HHMMSS> pattern, so callers can fall back to their
#     old naming scheme instead of crashing on unexpected filenames.
#     """
#     stem = Path(video_path).stem
#     match = _TIMESTAMP_RE.search(stem)
#     if not match:
#         return None

#     date_str, time_str = match.group(1), match.group(2)
#     try:
#         start_dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
#     except ValueError:
#         # Matched the digit pattern but it's not a real date/time (e.g.
#         # some other 14-digit number that isn't actually a timestamp).
#         return None

#     camera_id = stem[: match.start()].rstrip("_") or "camera"
#     return VideoTimestampInfo(camera_id=camera_id, start_datetime=start_dt)


"""
utils/video_timestamp.py
==========================

Two filename schemes are supported:

1. Legacy CCTV/DVR exports, camera ID + start time:
       A05401C4F5FC_20260715_010447.mp4
       ^camera_id   ^date     ^time (HHMMSS)

2. App recordings (app_version 2.0-expo+), start time + GPS + uuid prefix,
   with no separator between date and time:
       20260903012029._28.646947_77.320347_7f071d1a.mp4
       ^^^^^^^^^^^^^^ YYYYMMDDHHMMSS   ^lat       ^lon        ^uuid prefix

This module parses whichever one matches and computes the real wall-clock
timestamp of any given frame (start time + frame_number / fps), so saved
face images can be named after when they actually happened instead of a
generic "best_face.jpg" or frame-counter filename.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# --- Scheme 1: legacy CCTV, "<camera_id>_YYYYMMDD_HHMMSS" -------------------
# Matches an 8-digit date (YYYYMMDD) followed by an underscore and a
# 6-digit time (HHMMSS) anywhere in the filename stem.
_LEGACY_TIMESTAMP_RE = re.compile(r"(\d{8})_(\d{6})")

# --- Scheme 2: app export, "YYYYMMDDHHMMSS._lat_lon_uuid" -------------------
# 14 contiguous digits (date+time, no separator), then "._", then signed
# decimal lat/lon, then a hex uuid prefix.
_APP_TIMESTAMP_RE = re.compile(
    r"^(?P<ts>\d{14})\._?(?P<lat>-?\d+\.\d+)_(?P<lon>-?\d+\.\d+)_(?P<uid>[0-9a-fA-F]+)$"
)


@dataclass
class VideoTimestampInfo:
    start_datetime: datetime           # recording start time parsed from the filename
    camera_id: Optional[str] = None    # legacy scheme only
    latitude: Optional[float] = None   # app scheme only
    longitude: Optional[float] = None  # app scheme only
    recording_uuid_prefix: Optional[str] = None  # app scheme only

    def frame_datetime(self, frame_number: int, fps: float) -> datetime:
        """Wall-clock time of a given frame, assuming constant fps from start_datetime."""
        if fps <= 0:
            fps = 25.0  # defensive fallback; shouldn't normally happen
        return self.start_datetime + timedelta(seconds=frame_number / fps)

    def frame_timestamp_str(self, frame_number: int, fps: float) -> str:
        """Full ISO-style timestamp with milliseconds, filename-safe
        (colons replaced with '-'), e.g. '2026-09-03T01-20-29.807'.
        Computed as start_time + frame_number/fps using real datetime
        arithmetic, so seconds/minutes/hours correctly roll over past 59
        into the next minute/hour."""
        dt = self.frame_datetime(frame_number, fps)
        iso = dt.isoformat(timespec="milliseconds")
        return iso.replace(":", "-")


def _parse_app_scheme(stem: str) -> Optional[VideoTimestampInfo]:
    match = _APP_TIMESTAMP_RE.match(stem)
    if not match:
        return None
    try:
        start_dt = datetime.strptime(match.group("ts"), "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return VideoTimestampInfo(
        start_datetime=start_dt,
        latitude=float(match.group("lat")),
        longitude=float(match.group("lon")),
        recording_uuid_prefix=match.group("uid"),
    )


def _parse_legacy_scheme(stem: str) -> Optional[VideoTimestampInfo]:
    match = _LEGACY_TIMESTAMP_RE.search(stem)
    if not match:
        return None
    date_str, time_str = match.group(1), match.group(2)
    try:
        start_dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
    except ValueError:
        # Matched the digit pattern but it's not a real date/time (e.g.
        # some other 14-digit number that isn't actually a timestamp).
        return None
    camera_id = stem[: match.start()].rstrip("_") or "camera"
    return VideoTimestampInfo(start_datetime=start_dt, camera_id=camera_id)


def parse_video_timestamp(video_path: str) -> Optional[VideoTimestampInfo]:
    """
    Parse the recording start time (and camera ID or GPS/uuid, depending
    on scheme) out of a video filename. Tries the new app-export scheme
    first, then falls back to the legacy CCTV scheme. Returns None if
    neither matches, so callers can fall back to their own naming scheme
    instead of crashing on unexpected filenames.
    """
    stem = Path(video_path).stem
    return _parse_app_scheme(stem) or _parse_legacy_scheme(stem)