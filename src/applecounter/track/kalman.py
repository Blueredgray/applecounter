"""Калмановский фильтр бокса и общая база для трекеров BoT-SORT / DeepSORT.

Состояние КФ: [u, v, w, h, du, dv, dw, dh] — центр, ширина, высота и их
скорости в мировых координатах (камера скомпенсирована стабилизацией).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import Config
from ..utils import geometry as geo
from . import appearance as app
from .tracker import Track, TrackState, TrackView, _assign


class KalmanBoxFilter:
    """КФ постоянной скорости для бокса (u, v, w, h)."""

    def __init__(self, std_pos: float = 1.0 / 20, std_vel: float = 1.0 / 160,
                 std_meas: float = 1.0 / 20):
        self.std_pos = float(std_pos)
        self.std_vel = float(std_vel)
        self.std_meas = float(std_meas)
        self.F = np.eye(8, dtype=np.float64)
        self.F[:4, 4:] = np.eye(4, dtype=np.float64)
        self.H = np.eye(4, 8, dtype=np.float64)

    # -- преобразования -------------------------------------------------
    @staticmethod
    def box_to_z(box: Sequence[float]) -> np.ndarray:
        x1, y1, x2, y2 = (float(v) for v in box)
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0,
                         max(x2 - x1, 1.0), max(y2 - y1, 1.0)], dtype=np.float64)

    @staticmethod
    def z_to_box(z: Sequence[float]) -> np.ndarray:
        u, v, w, h = (float(x) for x in z)
        return np.array([u - w / 2.0, v - h / 2.0, u + w / 2.0, v + h / 2.0], dtype=np.float32)

    # -- шумовые матрицы -------------------------------------------------
    def _process_noise(self, w: float, h: float) -> np.ndarray:
        sp = [self.std_pos * h, self.std_pos * h, self.std_pos * w, self.std_pos * h]
        sv = [self.std_vel * h, self.std_vel * h, self.std_vel * w, self.std_vel * h]
        return np.diag(np.square(sp + sv))

    def _meas_noise(self, w: float, h: float) -> np.ndarray:
        rm = [self.std_meas * h, self.std_meas * h, self.std_meas * w, self.std_meas * h]
        return np.diag(np.square(rm))

    # -- шаги --------------------------------------------------------------
    def initiate(self, box: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        z = self.box_to_z(box)
        mean = np.zeros(8, dtype=np.float64)
        mean[:4] = z
        cov = np.diag(np.square(
            [self.std_pos * z[3], self.std_pos * z[3], self.std_pos * z[2], self.std_pos * z[3]]
            + [self.std_vel * z[3] * 10, self.std_vel * z[3] * 10,
               self.std_vel * z[2] * 10, self.std_vel * z[3] * 10]))
        return mean, cov

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.F @ mean
        h = max(float(mean[3]), 1.0)
        w = max(float(mean[2]), 1.0)
        cov = self.F @ cov @ self.F.T + self._process_noise(w, h)
        return mean, cov

    def update(self, mean: np.ndarray, cov: np.ndarray, box: Sequence[float]
               ) -> Tuple[np.ndarray, np.ndarray]:
        z = self.box_to_z(box)
        h = max(float(mean[3]), 1.0)
        w = max(float(mean[2]), 1.0)
        innov = z - self.H @ mean
        s = self.H @ cov @ self.H.T + self._meas_noise(w, h)
        k = cov @ self.H.T @ np.linalg.inv(s)
        mean = mean + k @ innov
        cov = (np.eye(8) - k @ self.H) @ cov
        return mean, cov


class KFTrackerBase(ABC):
    """Общая обвязка: КФ-предсказание, сопоставление (субкласс), bookkeeping.

    Как и MultiObjectTracker, работает в мировых координатах (движение камеры
    скомпенсировано стабилизацией) и отдаёт TrackView с count_event один раз
    при подтверждении трека.
    """

    name = "kf"
    coordinate_system = "world"

    def __init__(self, cfg: Config):
        t = cfg.get("tracker", {}) or {}
        self.high_conf = float(t.get("high_conf", 0.45))
        self.low_conf = float(t.get("low_conf", 0.15))
        self.min_iou = float(t.get("min_iou", 0.25))
        self.max_age = int(t.get("max_age", 25))
        self.reid_max_age = int(t.get("reid_max_age", 60))
        self.n_init = int(t.get("n_init", 2))
        self.use_appearance = bool(t.get("use_appearance", True))
        self.appearance_gate = float(t.get("appearance_gate", 0.55))
        self.reid_max_dist = float(t.get("reid_max_dist", 120.0))
        self.velocity_damping = float(t.get("velocity_damping", 0.7))

        self.kf = KalmanBoxFilter()
        self._tracks: Dict[int, Track] = {}
        self._archive: Dict[int, Track] = {}
        self._state: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._next_id = 1
        self.n_id_switches = 0
        self.n_created = 0

    @property
    def tracks(self) -> Dict[int, Track]:
        return {**self._archive, **self._tracks}

    def active_tracks(self) -> List[Track]:
        return [t for t in self._tracks.values() if t.active]

    # -- предсказание/измерение -------------------------------------------
    def _predict_all(self, active: Sequence[Track]) -> Dict[int, np.ndarray]:
        pred_boxes: Dict[int, np.ndarray] = {}
        for t in active:
            mean, cov = self._state[t.tid]
            mean, cov = self.kf.predict(mean, cov)
            self._state[t.tid] = (mean, cov)
            pred_boxes[t.tid] = self.kf.z_to_box(mean[:4])
        return pred_boxes

    @abstractmethod
    def _associate(self, active: Sequence[Track], dets: np.ndarray,
                   det_hists: Sequence[Optional[np.ndarray]],
                   pred_boxes: Dict[int, np.ndarray],
                   high_mask: np.ndarray, low_mask: np.ndarray) -> Dict[int, int]:
        """Вернуть сопоставления {tid: индекс детекции}."""

    # -- основной шаг -------------------------------------------------------
    def update(self, frame_idx: int, detections: np.ndarray, image: Optional[np.ndarray] = None,
               offset: Tuple[float, float] = (0.0, 0.0)) -> List[TrackView]:
        dets = np.asarray(detections, dtype=np.float32).reshape(-1, 6)
        det_hists: List[Optional[np.ndarray]] = []
        if image is not None and len(dets) and self.use_appearance:
            screen = geo.shift(dets[:, :4], float(offset[0]), float(offset[1]))
            det_hists = [app.box_histogram(image, box) for box in screen]
        else:
            det_hists = [None] * len(dets)

        active = self.active_tracks()
        pred_boxes = self._predict_all(active)

        high_mask = dets[:, 4] >= self.high_conf if len(dets) else np.zeros(0, dtype=bool)
        low_mask = (dets[:, 4] >= self.low_conf) & ~high_mask if len(dets) else np.zeros(0, dtype=bool)
        det_for_track = self._associate(active, dets, det_hists, pred_boxes, high_mask, low_mask)
        matched = set(det_for_track.values())

        views: List[TrackView] = []
        count_events: set = set()
        for tid, det_idx in det_for_track.items():
            track = self._tracks[tid]
            box = dets[det_idx, :4]
            score = float(dets[det_idx, 4])
            hist = det_hists[det_idx]
            # измерение корректирует КФ
            mean, cov = self._state[tid]
            self._state[tid] = self.kf.update(mean, cov, box)
            track.update(box, score, hist, frame_idx, damping=self.velocity_damping)
            if track.state == TrackState.CONFIRMED and not track.counted:
                track.counted = True
                count_events.add(tid)
            views.append(TrackView(track_id=tid, box=tuple(float(v) for v in track.box),
                                   conf=score, state=track.state, hits=track.hits,
                                   count_event=tid in count_events, updated=True))

        for det_idx in range(len(dets)):
            if det_idx in matched or dets[det_idx, 4] < self.high_conf:
                continue
            track = Track(tid=self._next_id, box=dets[det_idx, :4],
                          score=float(dets[det_idx, 4]), hist=det_hists[det_idx],
                          frame_idx=frame_idx)
            track._n_init = self.n_init
            if self.n_init <= 1:
                track.state = TrackState.CONFIRMED
                track.counted = True
                count_events.add(track.tid)
            self._tracks[track.tid] = track
            self._state[track.tid] = self.kf.initiate(dets[det_idx, :4])
            self._next_id += 1
            self.n_created += 1
            views.append(TrackView(track_id=track.tid, box=tuple(float(v) for v in track.box),
                                   conf=track.score, state=track.state, hits=track.hits,
                                   count_event=track.tid in count_events, updated=True))

        for track in active:
            if track.tid in det_for_track:
                continue
            track.age += 1
            track.time_since_update += 1
            track.mark_missed(self.max_age, self.reid_max_age)
            if track.state != TrackState.REMOVED:
                views.append(TrackView(track_id=track.tid,
                                       box=tuple(float(v) for v in pred_boxes[track.tid]),
                                       conf=track.score, state=track.state, hits=track.hits,
                                       count_event=False, updated=False))

        removed = {tid: t for tid, t in self._tracks.items() if t.state == TrackState.REMOVED}
        self._archive.update(removed)
        self._tracks = {tid: t for tid, t in self._tracks.items() if t.state != TrackState.REMOVED}
        self._state = {tid: s for tid, s in self._state.items() if tid in self._tracks}
        return views
