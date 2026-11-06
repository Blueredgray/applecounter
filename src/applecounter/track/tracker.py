"""Множественный трекинг объектов (яблок) в мировых (стабилизированных) координатах.

Ключевые идеи против повторного счёта:
  * детекции переводятся в мировые координаты (компенсация движения камеры), поэтому
    один и тот же плод остаётся в одном месте — IoU-ассоциация работает между кадрами;
  * дедупликация внутри кадра через NMS (см. детекторы);
  * каждому объекту присваивается глобальный уникальный id; счёт начисляется один раз
    при подтверждении трека (n_init последовательных совпадений);
  * после пропусков (окклюзии) трек восстанавливается по IoU+скорости, а при нулевом
    IoU — по внешнему виду (косинус цветовой гистограммы), id не меняется.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..config import Config
from ..utils import geometry as geo
from . import appearance as app

EMPTY = np.zeros((0, 6), dtype=np.float32)


class TrackState(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    REMOVED = "removed"


@dataclass
class TrackView:
    """Срез состояния трека на текущем кадре (для счёта и визуализации)."""

    track_id: int
    box: Tuple[float, float, float, float]
    conf: float
    state: TrackState
    hits: int
    count_event: bool  # True — трек только что подтверждён и начислен
    updated: bool


class Track:
    __slots__ = (
        "tid", "box", "score", "hist", "state", "hits", "age",
        "time_since_update", "velocity", "first_frame", "last_frame",
        "counted", "n_obs", "positions", "_n_init",
    )

    def __init__(self, tid: int, box: np.ndarray, score: float, hist: Optional[np.ndarray], frame_idx: int):
        self.tid = int(tid)
        self.box = np.asarray(box, dtype=np.float32)
        self.score = float(score)
        self.hist = hist
        self.state = TrackState.TENTATIVE
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.velocity = np.zeros(2, dtype=np.float32)
        self.first_frame = int(frame_idx)
        self.last_frame = int(frame_idx)
        self.counted = False
        self.n_obs = 1
        self._n_init = 2  # подтверждений до CONFIRMED (переопределяется трекером)
        self.positions: List[Tuple[int, np.ndarray]] = [(int(frame_idx), self.box.copy())]

    # -- предсказание ---------------------------------------------------
    def predict_box(self) -> np.ndarray:
        box = self.box.copy()
        box[0] += self.velocity[0]
        box[2] += self.velocity[0]
        box[1] += self.velocity[1]
        box[3] += self.velocity[1]
        return box

    # -- обновление -----------------------------------------------------
    def update(self, box: np.ndarray, score: float, hist: Optional[np.ndarray], frame_idx: int,
               damping: float = 0.7) -> None:
        box = np.asarray(box, dtype=np.float32)
        old_center = geo.center(self.box[None, :])[0]
        new_center = geo.center(box[None, :])[0]
        measured_velocity = new_center - old_center
        self.velocity = damping * self.velocity + (1.0 - damping) * measured_velocity
        self.box = box
        self.score = float(score)
        if hist is not None:
            self.hist = hist
        self.hits += 1
        self.age += 1
        self.time_since_update = 0
        self.last_frame = int(frame_idx)
        self.n_obs += 1
        self.positions.append((int(frame_idx), box.copy()))
        if self.state == TrackState.TENTATIVE and self.hits >= self._n_init:
            self.state = TrackState.CONFIRMED
        elif self.state in (TrackState.LOST, TrackState.CONFIRMED):
            self.state = TrackState.CONFIRMED

    def mark_missed(self, max_age: int, reid_max_age: int) -> None:
        if self.state == TrackState.TENTATIVE:
            # не подтверждённые треки живут до max_age — переживают краткие пропуски детекции
            if self.time_since_update > max_age:
                self.state = TrackState.REMOVED
            return
        if self.time_since_update > reid_max_age:
            self.state = TrackState.REMOVED
        elif self.time_since_update > max_age:
            self.state = TrackState.LOST

    @property
    def active(self) -> bool:
        return self.state in (TrackState.TENTATIVE, TrackState.CONFIRMED, TrackState.LOST)


def _assign(cost: np.ndarray, max_cost: float) -> List[Tuple[int, int]]:
    """Венгерский алгоритм с отсечением по стоимости."""
    if cost.size == 0:
        return []
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] <= max_cost]


class MultiObjectTracker:
    """Комбинированный трекер: IoU (высокая/низкая уверенность) + re-ID по внешнему виду."""

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
        # пространственный гейт re-ID: яблоко в мировых координатах статично,
        # поэтому далеко от прежнего положения трек перепривязываться не должен
        self.reid_max_dist = float(t.get("reid_max_dist", 120.0))
        self.velocity_damping = float(t.get("velocity_damping", 0.7))

        self._tracks: Dict[int, Track] = {}
        self._archive: Dict[int, Track] = {}
        self._next_id = 1
        self.n_id_switches = 0
        self.n_created = 0

    # -- утилиты --------------------------------------------------------
    @property
    def tracks(self) -> Dict[int, Track]:
        return {**self._archive, **self._tracks}

    def active_tracks(self) -> List[Track]:
        return [t for t in self._tracks.values() if t.active]

    def _cost_iou(self, tracks: Sequence[Track], dets: np.ndarray) -> np.ndarray:
        if not tracks or len(dets) == 0:
            return np.zeros((len(tracks), len(dets)), dtype=np.float32)
        pred = np.stack([t.predict_box() for t in tracks], axis=0)
        iou = geo.iou_matrix(pred, dets[:, :4])
        return (1.0 - iou).astype(np.float32)

    def _cost_appearance(self, tracks: Sequence[Track], dets: np.ndarray,
                         det_hists: Sequence[Optional[np.ndarray]]) -> np.ndarray:
        cost = np.ones((len(tracks), len(dets)), dtype=np.float32)
        det_centers = geo.center(dets[:, :4]) if len(dets) else np.zeros((0, 2), dtype=np.float32)
        for i, tr in enumerate(tracks):
            tr_center = geo.center(tr.box[None, :4])[0]
            for j, hist in enumerate(det_hists):
                if tr.hist is None or hist is None:
                    continue
                dist = float(np.linalg.norm(det_centers[j] - tr_center))
                if dist > self.reid_max_dist:
                    continue  # слишком далеко: тот же плод не мог «переехать»
                cost[i, j] = app.appearance_cost(tr.hist, hist)
        return cost

    # -- основной шаг ---------------------------------------------------
    def update(self, frame_idx: int, detections: np.ndarray, image: Optional[np.ndarray] = None,
               offset: Tuple[float, float] = (0.0, 0.0)) -> List[TrackView]:
        """Обработать кадр.

        detections — (N, 6) в мировых координатах; image — экранный кадр (для appearance);
        offset — накопленное смещение камеры (мир -> экран: screen = world + offset).
        """
        dets = np.asarray(detections, dtype=np.float32).reshape(-1, 6)
        det_hists: List[Optional[np.ndarray]] = []
        if image is not None and len(dets) and self.use_appearance:
            screen = geo.shift(dets[:, :4], float(offset[0]), float(offset[1]))
            det_hists = [app.box_histogram(image, box) for box in screen]
        else:
            det_hists = [None] * len(dets)

        active = self.active_tracks()
        high_mask = dets[:, 4] >= self.high_conf if len(dets) else np.zeros(0, dtype=bool)
        low_mask = (dets[:, 4] >= self.low_conf) & ~high_mask if len(dets) else np.zeros(0, dtype=bool)

        matched: set = set()
        det_for_track: Dict[int, int] = {}

        # Этап 1: IoU — все активные треки x детекции с высокой уверенностью
        stage1_tracks = list(active)
        stage1_dets = np.where(high_mask)[0]
        if stage1_tracks and len(stage1_dets):
            cost = self._cost_iou(stage1_tracks, dets[stage1_dets])
            for ti, di in _assign(cost, 1.0 - self.min_iou):
                det_idx = int(stage1_dets[di])
                det_for_track[stage1_tracks[ti].tid] = det_idx
                matched.add(det_idx)

        # Этап 2: оставшиеся треки x детекции с низкой уверенностью (заполнение пропусков)
        remaining_tracks = [t for t in active if t.tid not in det_for_track]
        stage2_dets = np.where(low_mask)[0]
        if remaining_tracks and len(stage2_dets):
            cost = self._cost_iou(remaining_tracks, dets[stage2_dets])
            for ti, di in _assign(cost, 1.0 - self.min_iou):
                det_idx = int(stage2_dets[di])
                det_for_track[remaining_tracks[ti].tid] = det_idx
                matched.add(det_idx)

        # Этап 3: re-ID потерянных треков по внешнему виду (IoU == 0 после окклюзии/сдвига)
        if self.use_appearance:
            lost_tracks = [
                t for t in active
                if t.tid not in det_for_track and t.state == TrackState.LOST and t.hist is not None
            ]
            free_dets = np.array([i for i in range(len(dets)) if i not in matched], dtype=int)
            if lost_tracks and len(free_dets):
                cost = self._cost_appearance(lost_tracks, dets[free_dets], [det_hists[i] for i in free_dets])
                for ti, di in _assign(cost, 1.0 - self.appearance_gate):
                    det_idx = int(free_dets[di])
                    det_for_track[lost_tracks[ti].tid] = det_idx
                    matched.add(det_idx)

        # -- обновление сопоставленных треков ----------------------------
        views: List[TrackView] = []
        count_events: set = set()
        for tid, det_idx in det_for_track.items():
            track = self._tracks[tid]
            box = dets[det_idx, :4]
            score = float(dets[det_idx, 4])
            hist = det_hists[det_idx]
            track.update(box, score, hist, frame_idx, damping=self.velocity_damping)
            if track.state == TrackState.CONFIRMED and not track.counted:
                track.counted = True
                count_events.add(tid)
            views.append(
                TrackView(
                    track_id=tid,
                    box=tuple(float(v) for v in track.box),
                    conf=score,
                    state=track.state,
                    hits=track.hits,
                    count_event=tid in count_events,
                    updated=True,
                )
            )

        # -- новые треки из несопоставленных высоконадёжных детекций ------
        for det_idx in range(len(dets)):
            if det_idx in matched:
                continue
            if dets[det_idx, 4] < self.high_conf:
                continue
            track = Track(
                tid=self._next_id,
                box=dets[det_idx, :4],
                score=float(dets[det_idx, 4]),
                hist=det_hists[det_idx],
                frame_idx=frame_idx,
            )
            track._n_init = self.n_init
            if self.n_init <= 1:
                track.state = TrackState.CONFIRMED
                track.counted = True
                count_events.add(track.tid)
            self._tracks[track.tid] = track
            self._next_id += 1
            self.n_created += 1
            views.append(
                TrackView(
                    track_id=track.tid,
                    box=tuple(float(v) for v in track.box),
                    conf=track.score,
                    state=track.state,
                    hits=track.hits,
                    count_event=track.tid in count_events,
                    updated=True,
                )
            )

        # -- пропущенные треки -------------------------------------------
        for track in active:
            if track.tid in det_for_track:
                continue
            track.age += 1
            track.time_since_update += 1
            track.mark_missed(self.max_age, self.reid_max_age)
            if track.state != TrackState.REMOVED:
                views.append(
                    TrackView(
                        track_id=track.tid,
                        box=tuple(float(v) for v in track.predict_box()),
                        conf=track.score,
                        state=track.state,
                        hits=track.hits,
                        count_event=False,
                        updated=False,
                    )
                )

        # Keep removed tracks for final assignment and audit; only active tracking state is pruned.
        removed = {tid: t for tid, t in self._tracks.items() if t.state == TrackState.REMOVED}
        self._archive.update(removed)
        self._tracks = {tid: t for tid, t in self._tracks.items() if t.state != TrackState.REMOVED}
        return views
