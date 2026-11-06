"""Adapters for established BoT-SORT, ByteTrack and DeepSORT implementations.

These trackers operate in screen coordinates. BoT-SORT performs its own camera
motion compensation; DeepSORT uses its matching cascade, Mahalanobis gate and a
CNN appearance embedder. No external world-coordinate stabilizer is applied.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .tracker import Track, TrackState, TrackView


class ExternalTracker:
    """Common TrackView interface for third-party tracking-by-detection engines."""

    coordinate_system = "screen"

    def __init__(self, cfg, name: str):
        self.cfg = cfg
        self.name = name
        t = cfg.get("tracker", {}) or {}
        self.n_init = int(t.get("n_init", 2))
        self._tracks: Dict[int, Track] = {}
        self._id_map: Dict[str, int] = {}
        self._next_synthetic_id = 1
        self.n_created = 0
        self.n_id_switches = 0
        self.implementation = ""

    @property
    def tracks(self) -> Dict[int, Track]:
        return self._tracks

    def _numeric_id(self, raw_id) -> int:
        try:
            return int(raw_id)
        except (TypeError, ValueError):
            key = str(raw_id)
            if key not in self._id_map:
                self._id_map[key] = self._next_synthetic_id
                self._next_synthetic_id += 1
            return self._id_map[key]

    def _view(self, raw_id, box, score: float, frame_idx: int) -> TrackView:
        tid = self._numeric_id(raw_id)
        box = np.asarray(box, dtype=np.float32).reshape(4)
        track = self._tracks.get(tid)
        created = track is None
        if created:
            track = Track(tid=tid, box=box, score=float(score), hist=None, frame_idx=frame_idx)
            # Count only observations actually emitted by the external engine.
            # Its private tentative hits are not observable through this adapter.
            track.hits = 1
            track.n_obs = 1
            track.state = TrackState.CONFIRMED
            track.counted = True
            self._tracks[tid] = track
            self.n_created += 1
        else:
            track.update(box, float(score), None, frame_idx)
        return TrackView(
            track_id=tid,
            box=tuple(float(v) for v in box),
            conf=float(score),
            state=TrackState.CONFIRMED,
            hits=track.hits,
            count_event=created,
            updated=True,
        )

    def close(self) -> None:
        return None


class BoxMotTracker(ExternalTracker):
    """BoxMOT adapter; packed output is x1,y1,x2,y2,id,conf,cls,..."""

    def __init__(self, cfg, name: str):
        super().__init__(cfg, name)
        try:
            from boxmot import BotSort, ByteTrack
            from boxmot.trackers.botsort.config import BotSortConfig
            from boxmot.trackers.bytetrack.config import ByteTrackConfig
        except ImportError as exc:
            raise ImportError(f"{name} requires the 'boxmot' package") from exc
        t = cfg.get("tracker", {}) or {}
        common = dict(
            max_age=int(t.get("max_age", 25)),
            min_hits=int(t.get("n_init", 2)),
            iou_threshold=float(t.get("min_iou", 0.25)),
        )
        if name == "botsort":
            config = BotSortConfig(
                **common,
                track_high_thresh=float(t.get("high_conf", 0.45)),
                track_low_thresh=float(t.get("low_conf", 0.15)),
                new_track_thresh=float(t.get("high_conf", 0.45)),
                det_thresh=float(t.get("low_conf", 0.15)),
                use_cmc=bool(t.get("cmc_enabled", True)),
                cmc_method=str(t.get("cmc_method", "sof")),
                use_embeddings=bool(t.get("use_embeddings", True)),
                appearance_thresh=float(t.get("appearance_gate", 0.55)),
            )
            self.tracker = BotSort(config=config)
            self.implementation = "boxmot.BotSort"
        elif name == "bytetrack":
            config = ByteTrackConfig(
                **common,
                track_thresh=float(t.get("high_conf", 0.45)),
                min_conf=float(t.get("low_conf", 0.15)),
                use_cmc=bool(t.get("cmc_enabled", False)),
                cmc_method=str(t.get("cmc_method", "sof")),
            )
            self.tracker = ByteTrack(config=config)
            self.implementation = "boxmot.ByteTrack"
        else:
            raise ValueError(name)

    def update(self, frame_idx: int, detections, image=None, offset=(0.0, 0.0)):
        dets = np.asarray(detections, dtype=np.float32).reshape(-1, 6)
        output = self.tracker.update(dets, image)
        if output is None:
            return []
        output = np.asarray(output)
        if output.size == 0:
            return []
        output = output.reshape(len(output), -1)
        views = []
        for row in output:
            if len(row) < 6:
                continue
            views.append(self._view(row[4], row[:4], float(row[5]), frame_idx))
        return views


class DeepSortRealtimeTracker(ExternalTracker):
    """DeepSORT with CNN embeddings from the maintained deep-sort-realtime package."""

    def __init__(self, cfg):
        super().__init__(cfg, "deepsort")
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ImportError as exc:
            raise ImportError("DeepSORT requires the 'deep-sort-realtime' package") from exc
        t = cfg.get("tracker", {}) or {}
        device = str(cfg.get("project.device", "auto")).lower()
        self.tracker = DeepSort(
            max_iou_distance=float(t.get("max_iou_distance", 0.7)),
            max_age=int(t.get("max_age", 25)),
            n_init=int(t.get("n_init", 2)),
            max_cosine_distance=float(t.get("max_cosine_distance", 0.2)),
            nn_budget=t.get("nn_budget", 100),
            embedder=str(t.get("embedder", "mobilenet")),
            embedder_gpu=device != "cpu",
            half=bool(t.get("embedder_half", True)),
            bgr=True,
        )
        self.implementation = "deep_sort_realtime.DeepSort"

    def update(self, frame_idx: int, detections, image=None, offset=(0.0, 0.0)):
        if image is None:
            raise ValueError("DeepSORT requires the current image for CNN appearance embeddings")
        dets = np.asarray(detections, dtype=np.float32).reshape(-1, 6)
        high = float(self.cfg.get("tracker.high_conf", 0.45))
        raw = []
        for row in dets:
            if float(row[4]) < high:
                continue
            x1, y1, x2, y2 = map(float, row[:4])
            raw.append(([x1, y1, x2 - x1, y2 - y1], float(row[4]), int(row[5])))
        tracks = self.tracker.update_tracks(raw, frame=image)
        views = []
        for item in tracks:
            if not item.is_confirmed() or int(item.time_since_update) != 0:
                continue
            score = float(item.det_conf) if item.det_conf is not None else 0.0
            views.append(self._view(item.track_id, item.to_ltrb(), score, frame_idx))
        return views

    def close(self) -> None:
        self.tracker.delete_all_tracks()
