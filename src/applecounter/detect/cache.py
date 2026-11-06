"""Versioned per-frame detection cache.

The cache stores detections in screen coordinates after the configured video
transformations. Tracking experiments can therefore share exactly the same
detector output instead of running YOLO once per tracker.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

import numpy as np

from .base import EMPTY_DETECTIONS

FORMAT_VERSION = 1


class DetectionCache:
    """In-memory builder and reader for a compressed ``.npz`` detection cache."""

    def __init__(self, metadata: Optional[dict] = None):
        self.metadata = dict(metadata or {})
        self._frames: list[int] = []
        self._by_frame: Dict[int, np.ndarray] = {}

    def add(self, frame_idx: int, detections: np.ndarray) -> None:
        frame_idx = int(frame_idx)
        if frame_idx in self._by_frame:
            raise ValueError(f"frame {frame_idx} already exists in detection cache")
        det = np.asarray(detections, dtype=np.float32).reshape(-1, 6)
        self._frames.append(frame_idx)
        self._by_frame[frame_idx] = det.copy()

    def get(self, frame_idx: int) -> np.ndarray:
        return self._by_frame.get(int(frame_idx), EMPTY_DETECTIONS).copy()

    def __contains__(self, frame_idx: int) -> bool:
        return int(frame_idx) in self._by_frame

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self):
        return iter(self._frames)

    def items(self) -> Iterator[tuple[int, np.ndarray]]:
        for frame in self._frames:
            yield frame, self.get(frame)

    def save(self, path: str | Path) -> Path:
        if not self._frames:
            raise ValueError("refusing to save an empty detection cache")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        order = np.argsort(np.asarray(self._frames, dtype=np.int64))
        frames = np.asarray(self._frames, dtype=np.int64)[order]
        arrays = [self._by_frame[int(f)] for f in frames]
        lengths = np.asarray([len(a) for a in arrays], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        flat = (np.concatenate(arrays, axis=0) if arrays else EMPTY_DETECTIONS).astype(np.float32)
        metadata = dict(self.metadata)
        metadata["format_version"] = FORMAT_VERSION
        metadata["n_frames"] = len(frames)
        metadata["n_detections"] = int(len(flat))
        tmp = path.with_suffix(path.suffix + ".tmp.npz")
        np.savez_compressed(
            tmp,
            frames=frames,
            offsets=offsets,
            detections=flat,
            metadata=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
        )
        tmp.replace(path)
        self.metadata = metadata
        return path

    @classmethod
    def load(cls, path: str | Path, expected_metadata: Optional[dict] = None) -> "DetectionCache":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            required = {"frames", "offsets", "detections", "metadata"}
            if not required.issubset(data.files):
                raise ValueError(f"invalid detection cache {path}: missing {sorted(required - set(data.files))}")
            metadata = json.loads(str(data["metadata"].item()))
            if int(metadata.get("format_version", -1)) != FORMAT_VERSION:
                raise ValueError(f"unsupported detection cache version in {path}")
            frames = data["frames"].astype(np.int64)
            offsets = data["offsets"].astype(np.int64)
            flat = data["detections"].astype(np.float32).reshape(-1, 6)
        if len(offsets) != len(frames) + 1 or offsets[0] != 0 or offsets[-1] != len(flat):
            raise ValueError(f"corrupt detection cache offsets in {path}")
        if len(np.unique(frames)) != len(frames):
            raise ValueError(f"duplicate frame indices in {path}")
        cache = cls(metadata)
        for i, frame in enumerate(frames):
            cache.add(int(frame), flat[offsets[i]:offsets[i + 1]])
        if expected_metadata:
            mismatches = {
                key: {"cache": metadata.get(key), "expected": value}
                for key, value in expected_metadata.items()
                if metadata.get(key) != value
            }
            if mismatches:
                raise ValueError(f"detection cache configuration mismatch in {path}: {mismatches}")
        return cache
