"""BoT-SORT (Zhang et al., 2022) для подсчёта яблок.

BoT-SORT = ассоциация в два этапа (высоконадёжные / низконадёжные детекции,
как в ByteTrack) + Калмановский motion model + этап re-ID по внешнему виду
для непосопоставленных треков. Компенсация движения камеры (CMX) в нашей
реализации выполняется заранее стабилизацией: детекции подаются в мировых
координатах.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from ..utils import geometry as geo
from . import appearance as app
from .kalman import KFTrackerBase
from .tracker import Track, _assign


class BotSortTracker(KFTrackerBase):
    """BoT-SORT: КФ + two-stage (high/low conf) + re-ID cosine-гейт."""

    name = "botsort"

    def _associate(self, active: Sequence[Track], dets: np.ndarray,
                   det_hists: Sequence[Optional[np.ndarray]],
                   pred_boxes: Dict[int, np.ndarray],
                   high_mask: np.ndarray, low_mask: np.ndarray) -> Dict[int, int]:
        matched: Dict[int, int] = {}
        used_dets: set = set()

        def _iou_stage(tracks: Sequence[Track], det_idx: np.ndarray) -> None:
            if not tracks or len(det_idx) == 0:
                return
            pred = np.stack([pred_boxes[t.tid] for t in tracks], axis=0)
            iou = geo.iou_matrix(pred, dets[det_idx, :4])
            for ti, di in _assign((1.0 - iou).astype(np.float32), 1.0 - self.min_iou):
                det_i = int(det_idx[di])
                matched[tracks[ti].tid] = det_i
                used_dets.add(det_i)

        # Этап 1: IoU — активные треки x высоконадёжные детекции
        _iou_stage(list(active), np.where(high_mask)[0])

        # Этап 2: IoU — оставшиеся треки x низконадёжные детекции (заполнение
        # пропусков при окклюзии, основная идея ByteTrack внутри BoT-SORT)
        remaining = [t for t in active if t.tid not in matched]
        _iou_stage(remaining, np.where(low_mask)[0])

        # Этап 3: re-ID по внешнему виду — оставшиеся треки x свободные детекции
        if not self.use_appearance:
            return matched
        remaining = [t for t in active if t.tid not in matched]
        free = np.array([i for i in range(len(dets)) if i not in used_dets], dtype=int)
        if remaining and len(free):
            det_centers = geo.center(dets[free, :4])
            cost = np.ones((len(remaining), len(free)), dtype=np.float32)
            for i, tr in enumerate(remaining):
                tr_center = geo.center(pred_boxes[tr.tid].reshape(1, 4))[0]
                for j, det_j in enumerate(free):
                    hist = det_hists[det_j]
                    if tr.hist is None or hist is None:
                        continue
                    if float(np.linalg.norm(det_centers[j] - tr_center)) > self.reid_max_dist:
                        continue
                    cost[i, j] = app.appearance_cost(tr.hist, hist)
            for ti, di in _assign(cost, 1.0 - self.appearance_gate):
                det_i = int(free[di])
                matched[remaining[ti].tid] = det_i
                used_dets.add(det_i)
        return matched
