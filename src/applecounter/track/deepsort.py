"""Лёгкий DeepSORT-подобный baseline с HSV-признаками, не эталонный DeepSORT.

Сохранён для быстрых unit-тестов и отладки. Для исследовательских сравнений
используйте ``tracker.name=deepsort`` — CNN-реализацию deep-sort-realtime.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from ..utils import geometry as geo
from . import appearance as app
from .kalman import KFTrackerBase
from .tracker import Track, _assign

GALLERY_SIZE = 8  # сколько последних признаков хранит галерея трека


class DeepSortTracker(KFTrackerBase):
    """DeepSORT: КФ + каскадная ассоциация (движение -> ReID-галерея)."""

    name = "deepsort_lite"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._gallery: Dict[int, List[np.ndarray]] = {}

    # -- галерея признаков --------------------------------------------------
    def _gallery_mean(self, tid: int) -> Optional[np.ndarray]:
        feats = self._gallery.get(tid)
        if not feats:
            return None
        mean = np.mean(np.stack(feats, axis=0), axis=0)
        norm = float(np.linalg.norm(mean))
        return mean / norm if norm > 0 else mean

    def _add_feature(self, tid: int, hist: Optional[np.ndarray]) -> None:
        if hist is None:
            return
        feats = self._gallery.setdefault(tid, [])
        feats.append(hist.astype(np.float32))
        if len(feats) > GALLERY_SIZE:
            feats.pop(0)

    # -- ассоциация ---------------------------------------------------------
    def _associate(self, active: Sequence[Track], dets: np.ndarray,
                   det_hists: Sequence[Optional[np.ndarray]],
                   pred_boxes: Dict[int, np.ndarray],
                   high_mask: np.ndarray, low_mask: np.ndarray) -> Dict[int, int]:
        matched: Dict[int, int] = {}
        used_dets: set = set()
        hi = np.where(high_mask)[0]

        # Каскад 1: движение — IoU-гейт предсказанных боксов с high-conf детекциями
        if active and len(hi):
            pred = np.stack([pred_boxes[t.tid] for t in active], axis=0)
            iou = geo.iou_matrix(pred, dets[hi, :4])
            for ti, di in _assign((1.0 - iou).astype(np.float32), 1.0 - self.min_iou):
                det_i = int(hi[di])
                matched[active[ti].tid] = det_i
                used_dets.add(det_i)

        # Каскад 2: внешний вид — косинус к галерее трека (в т.ч. LOST-треки)
        if self.use_appearance:
            remaining = [t for t in active if t.tid not in matched]
            free = np.array([i for i in hi if i not in used_dets], dtype=int)
            if remaining and len(free):
                det_centers = geo.center(dets[free, :4])
                cost = np.ones((len(remaining), len(free)), dtype=np.float32)
                for i, tr in enumerate(remaining):
                    tr_center = geo.center(pred_boxes[tr.tid].reshape(1, 4))[0]
                    gallery = self._gallery_mean(tr.tid)
                    if gallery is None:
                        gallery = tr.hist
                    if gallery is None:
                        continue
                    for j, det_j in enumerate(free):
                        hist = det_hists[det_j]
                        if hist is None:
                            continue
                        if float(np.linalg.norm(det_centers[j] - tr_center)) > self.reid_max_dist:
                            continue
                        cost[i, j] = app.appearance_cost(gallery, hist)
                for ti, di in _assign(cost, 1.0 - self.appearance_gate):
                    det_i = int(free[di])
                    matched[remaining[ti].tid] = det_i
                    used_dets.add(det_i)
        return matched

    # -- обновление галереи ---------------------------------------------------
    def update(self, frame_idx: int, detections, image=None, offset=(0.0, 0.0)):
        views = super().update(frame_idx, detections, image=image, offset=offset)
        dets = np.asarray(detections, dtype=np.float32).reshape(-1, 6)
        if image is not None and len(dets) and self.use_appearance:
            screen = geo.shift(dets[:, :4], float(offset[0]), float(offset[1]))
            hists = {i: app.box_histogram(image, box) for i, box in enumerate(screen)}
            for view in views:
                if not view.updated:
                    continue
                # view.box and the cached detections are both in world coordinates.
                for i, box in enumerate(dets[:, :4]):
                    if np.allclose(box, np.asarray(view.box, dtype=np.float32), atol=1e-3):
                        self._add_feature(view.track_id, hists[i])
                        break
        self._gallery = {tid: f for tid, f in self._gallery.items() if tid in self._tracks}
        return views
