"""MOT Challenge ground-truth reader used by T1 and other MOT datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def read_mot_gt(path: str | Path, zero_based: bool = True) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[int, np.ndarray]]]:
    """Read ``gt.txt`` into per-frame detections and persistent track boxes.

    MOT columns are frame, id, x, y, w, h, mark, class, visibility. Rows with
    mark=0 are ignored. Frame indices are converted from 1-based MOT to 0-based
    pipeline indices by default.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "gt" / "gt.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 6:
        raise ValueError(f"invalid MOT file: {path}")
    if df.shape[1] >= 7:
        df = df[df[6] > 0]
    per_frame: Dict[int, np.ndarray] = {}
    tracks: Dict[int, Dict[int, np.ndarray]] = {}
    for row in df.itertuples(index=False, name=None):
        frame = int(row[0]) - (1 if zero_based else 0)
        track_id = int(row[1])
        x, y, w, h = map(float, row[2:6])
        if w <= 0 or h <= 0:
            continue
        box = np.asarray([x, y, x + w, y + h], dtype=np.float32)
        det = np.asarray([*box, 1.0, 0.0], dtype=np.float32)
        per_frame.setdefault(frame, []).append(det)
        tracks.setdefault(frame, {})[track_id] = box
    per_frame = {
        frame: np.asarray(rows, dtype=np.float32).reshape(-1, 6)
        for frame, rows in per_frame.items()
    }
    return per_frame, tracks


def mot_identity_stats(gt_frames: Dict[int, Dict[int, np.ndarray]]) -> dict:
    appearances: Dict[int, int] = {}
    for frame in gt_frames.values():
        for track_id in frame:
            appearances[track_id] = appearances.get(track_id, 0) + 1
    repeated = sum(1 for n in appearances.values() if n > 1)
    return {
        "n_ids": len(appearances),
        "n_repeated_ids": repeated,
        "n_observations": sum(appearances.values()),
        "tracking_gt_valid": bool(appearances) and repeated > 0,
    }
