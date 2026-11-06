"""VFC marker-based track-to-stretch assignment.

Follows Felip-Pomés et al.'s ``stretch_velocity``, ``get_change_frames`` and
``assign_all_apples`` procedure. Unlike the reference script, each track is
written once (majority vote if it is visible at several markers), and the
west-side reversal is applied to all tracks rather than dropping indirectly
assigned ones. These deliberate fixes are recorded in the run metadata.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


def _majority(values: list[int]) -> int:
    counts = Counter(values)
    # Stable tie-break: smaller stretch first, matching deterministic export.
    return min(counts, key=lambda value: (-counts[value], value))


def assign_vfc_stretches(
    observations: Mapping[int, Sequence[tuple[int, Sequence[float]]]],
    marker_frames: Sequence[int],
    image_width: float,
    video_name: str = "",
) -> tuple[Dict[int, int], dict]:
    """Return persistent ``track_id -> stretch`` using the VFC marker method.

    Boxes are in screen coordinates ``x1,y1,x2,y2`` and frames use the original
    video indexing. There are N markers and N-1 stretches. IDs outside 1..N-1
    are returned as 0 (unassigned).
    """
    markers = [int(value) for value in marker_frames]
    if len(markers) < 2 or any(b <= a for a, b in zip(markers, markers[1:])):
        raise ValueError("VFC marker frames must be strictly increasing")
    half = float(image_width) / 2.0
    tracks: Dict[int, list[tuple[int, np.ndarray]]] = {}
    for raw_id, rows in observations.items():
        tid = int(raw_id)
        tracks[tid] = sorted(
            (int(frame), np.asarray(box, dtype=float).reshape(4)) for frame, box in rows
        )
    centers = {
        tid: np.asarray([(box[0] + box[2]) / 2.0 for _, box in rows], dtype=float)
        for tid, rows in tracks.items()
    }
    frames = {tid: np.asarray([frame for frame, _ in rows], dtype=int) for tid, rows in tracks.items()}
    by_frame: Dict[int, list[int]] = defaultdict(list)
    for tid, rows in tracks.items():
        for frame, _ in rows:
            by_frame[frame].append(tid)

    direct_votes: Dict[int, list[int]] = defaultdict(list)
    velocities = []
    for index, marker in enumerate(markers):
        marker_stretch = index + 1
        present = by_frame.get(marker, [])
        local_velocities = []
        for tid in present:
            center = float(centers[tid][np.where(frames[tid] == marker)[0][0]])
            direct_votes[tid].append(marker_stretch - 1 if center <= half else marker_stretch)
            f = frames[tid]
            if len(f) >= 2 and int(f[-1] - f[0]) > 0:
                c = centers[tid]
                local_velocities.append(float(np.max(c) - np.min(c)) / float(f[-1] - f[0]))
        velocities.append(float(np.mean(local_velocities)) if local_velocities else 0.0)

    assigned: Dict[int, int] = {}
    for tid, votes in direct_votes.items():
        assigned[tid] = _majority(votes)

    change_frames = []
    for marker, velocity in zip(markers, velocities):
        index_stretch = len(change_frames) + 1
        if velocity == 0:
            change_frames.append((marker - 50, marker + 50, index_stretch))
        else:
            duration = int(float(image_width) / velocity)
            change_frames.append((int(marker - duration / 2), int(marker + duration / 2), index_stretch))

    for first, last, marker_stretch in change_frames:
        candidates: Dict[int, list[int]] = defaultdict(list)
        for tid, rows in tracks.items():
            if tid in assigned:
                continue
            for (frame, _), center in zip(rows, centers[tid]):
                if first < frame < last:
                    midpoint = (first + last) / 2.0
                    if frame < midpoint:
                        boundary_x = half + (midpoint - frame) * velocities[marker_stretch - 1]
                    else:
                        boundary_x = half - (frame - midpoint) * velocities[marker_stretch - 1]
                    candidates[tid].append(marker_stretch - 1 if center < boundary_x else marker_stretch)
        if velocities[marker_stretch - 1] != 0:
            for tid, votes in candidates.items():
                assigned[tid] = _majority(votes)
        # Reference fallback: every not-yet-assigned track seen before the end
        # of this transition belongs to the preceding stretch.
        for tid, f in frames.items():
            if tid not in assigned and len(f) and int(f[0]) < last:
                assigned[tid] = marker_stretch - 1

    n_stretches = len(markers) - 1
    assigned = {
        tid: (stretch if 1 <= stretch <= n_stretches else 0)
        for tid, stretch in assigned.items()
    }
    orientation = Path(video_name).stem.split("_")[4] if len(Path(video_name).stem.split("_")) > 4 else ""
    if orientation == "w":
        assigned = {tid: (n_stretches + 1 - stretch if stretch > 0 else 0)
                    for tid, stretch in assigned.items()}
    for tid in tracks:
        assigned.setdefault(tid, 0)
    meta = {
        "assignment": "vfc_reference",
        "image_width": float(image_width),
        "orientation": orientation,
        "marker_frames": markers,
        "stretch_velocities_px_per_frame": velocities,
        "change_frames": [list(row) for row in change_frames],
        "directly_assigned_tracks": len(direct_votes),
        "unassigned_tracks": sum(value == 0 for value in assigned.values()),
        "reference_fixes": ["one output per track", "west reversal for indirect assignments"],
    }
    return assigned, meta
