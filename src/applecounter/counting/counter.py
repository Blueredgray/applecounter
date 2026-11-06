"""Итоговый счёт: уникальные объекты, разбивка по деревьям, динамика по кадрам."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..track.tracker import TrackState, TrackView
from ..trees.segment import TreeMap
from ..utils.logging import get_logger

logger = get_logger()


@dataclass
class TrackRecord:
    """Итоговая карточка одного уникального объекта (яблока)."""

    track_id: int
    first_frame: int
    last_frame: int
    n_obs: int
    world_center: np.ndarray
    tree_id: int = 0
    mean_conf: float = 0.0

    @property
    def counted(self) -> bool:
        return self.n_obs > 0


@dataclass
class CountResult:
    total: int = 0
    per_tree: Dict[int, int] = field(default_factory=dict)
    unassigned: int = 0
    records: List[TrackRecord] = field(default_factory=list)
    per_frame_ids: List[int] = field(default_factory=list)   # уникальных объектов на кадр (суммарно)
    per_frame_active: List[int] = field(default_factory=list)  # активных треков на кадр

    def as_rows(self) -> List[dict]:
        return [
            {
                "track_id": r.track_id,
                "tree_id": r.tree_id,
                "first_frame": r.first_frame,
                "last_frame": r.last_frame,
                "n_obs": r.n_obs,
                "x": round(float(r.world_center[0]), 2),
                "y": round(float(r.world_center[1]), 2),
                "mean_conf": round(float(r.mean_conf), 3),
            }
            for r in self.records
        ]


class AppleCounter:
    """Считает каждый уникальный объект ровно один раз (по id трека)."""

    def __init__(self, min_obs: int = 2):
        self.min_obs = int(min_obs)
        self._records: Dict[int, TrackRecord] = {}
        self._conf_sum: Dict[int, float] = {}
        self._conf_n: Dict[int, int] = {}
        self._counted_ids: set = set()
        self.per_frame_ids: List[int] = []
        self.per_frame_active: List[int] = []
        self.n_double_count_events = 0  # контролирует, что повторов не было

    def reset(self) -> None:
        self._records.clear()
        self._conf_sum.clear()
        self._conf_n.clear()
        self._counted_ids.clear()
        self.per_frame_ids.clear()
        self.per_frame_active.clear()
        self.n_double_count_events = 0

    # -- шаг ------------------------------------------------------------
    def update(self, frame_idx: int, views: Sequence[TrackView]) -> None:
        active = 0
        created: set = set()
        for view in views:
            if view.state in (TrackState.CONFIRMED, TrackState.LOST, TrackState.TENTATIVE):
                active += 1
            if view.track_id in self._counted_ids:
                if view.count_event:
                    self.n_double_count_events += 1  # этого быть не должно
                continue
            if not view.count_event:
                continue

            box = np.asarray(view.box, dtype=np.float64)
            center = np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5])
            self._records[view.track_id] = TrackRecord(
                track_id=view.track_id,
                first_frame=frame_idx,
                last_frame=frame_idx,
                n_obs=1,
                world_center=center,
                mean_conf=float(view.conf),
            )
            self._conf_sum[view.track_id] = float(view.conf)
            self._conf_n[view.track_id] = 1
            self._counted_ids.add(view.track_id)
            created.add(view.track_id)
        # обновляем статистику наблюдений у уже учтённых треков
        for view in views:
            if view.track_id in created:
                continue  # запись создана в этом же кадре — наблюдение уже учтено
            rec = self._records.get(view.track_id)
            if rec is None:
                continue
            if view.updated:
                rec.n_obs += 1
                rec.last_frame = frame_idx
                box = np.asarray(view.box, dtype=np.float64)
                center = np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5])
                # скользящее среднее центра — устойчивое положение объекта
                k = rec.n_obs
                rec.world_center = rec.world_center + (center - rec.world_center) / min(k, 10)
                self._conf_sum[view.track_id] = self._conf_sum.get(view.track_id, 0.0) + float(view.conf)
                self._conf_n[view.track_id] = self._conf_n.get(view.track_id, 0) + 1

        self.per_frame_ids.append(len(self._counted_ids))
        self.per_frame_active.append(active)

    # -- финализация ----------------------------------------------------
    def finalize(self, tree_map: Optional[TreeMap] = None, tracks: Optional[dict] = None,
                 merge_dist: float = 14.0) -> CountResult:
        """Собрать итоги: уникальные объекты и разбивка по деревьям.

        tracks — словарь треков {track_id: Track} (история позиций и кадров).
        merge_dist — объединять треки, чьи мировые центры ближе merge_dist:
        осколки одного плода (фрагменты при окклюзии) дают несколько треков
        с одним и тем же положением, тогда как разные плоды разнесены дальше
        (типично >= 18-20 px для синтетики и существенно больше в поле).
        """
        records = [rec for rec in self._records.values() if rec.n_obs >= max(1, self.min_obs)]
        records.sort(key=lambda r: r.first_frame)

        for rec in records:
            n = self._conf_n.get(rec.track_id, 1)
            rec.mean_conf = self._conf_sum.get(rec.track_id, 0.0) / max(1, n)

        n_before_merge = len(records)
        records = self._merge_fragments(records, merge_dist)
        if len(records) != n_before_merge:
            logger.debug("слияние осколков треков: %d -> %d", n_before_merge, len(records))

        per_tree: Dict[int, int] = {}
        unassigned = 0
        if tree_map is not None:
            for rec in records:
                points, frames = self._track_geometry(rec, tracks)
                tree_id = tree_map.assign_track(points, frames)
                rec.tree_id = int(tree_id)
                if tree_id <= 0:
                    unassigned += 1
                else:
                    per_tree[tree_id] = per_tree.get(tree_id, 0) + 1
        else:
            per_tree = {0: len(records)}

        return CountResult(
            total=len(records),
            per_tree=dict(sorted(per_tree.items())),
            unassigned=unassigned,
            records=records,
            per_frame_ids=list(self.per_frame_ids),
            per_frame_active=list(self.per_frame_active),
        )

    @staticmethod
    def _merge_fragments(records: List["TrackRecord"], merge_dist: float) -> List["TrackRecord"]:
        """Объединить осколки одного объекта (ближе merge_dist в мировых координатах)."""
        if merge_dist <= 0 or len(records) < 2:
            return records
        reps: List[list] = []  # [record, position]
        for rec in sorted(records, key=lambda r: -r.n_obs):  # сильные треки — «якоря»
            center = np.asarray(rec.world_center, dtype=np.float64)
            target = None
            for rep in reps:
                if float(np.linalg.norm(rep[1] - center)) <= merge_dist:
                    target = rep
                    break
            if target is None:
                reps.append([rec, center.copy()])
                continue
            keep = target[0]
            keep.first_frame = min(keep.first_frame, rec.first_frame)
            keep.last_frame = max(keep.last_frame, rec.last_frame)
            keep.n_obs += rec.n_obs
            keep.mean_conf = max(keep.mean_conf, rec.mean_conf)
        merged = [rep[0] for rep in reps]
        merged.sort(key=lambda r: r.first_frame)
        return merged

    @staticmethod
    def _track_geometry(rec: TrackRecord, tracks: Optional[dict]):
        """Мировые точки и кадры трека (с запасными вариантами без истории)."""
        track = (tracks or {}).get(rec.track_id)
        if track is not None and getattr(track, "positions", None):
            frames = [f for f, _ in track.positions]
            points = np.array(
                [[(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5] for _, b in track.positions],
                dtype=np.float64,
            )
            return points, frames
        return rec.world_center.reshape(1, 2), [rec.first_frame]

