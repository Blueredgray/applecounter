"""Отрисовка детекций и треков на кадрах."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..track.tracker import TrackState, TrackView
from ..utils import geometry as geo

_STATE_COLOR = {
    TrackState.CONFIRMED: (60, 220, 60),
    TrackState.TENTATIVE: (200, 180, 60),
    TrackState.LOST: (60, 160, 240),
}


def color_for_id(track_id: int) -> Tuple[int, int, int]:
    """Стабильный цвет по id (HSV -> BGR)."""
    hue = int(track_id * 47) % 180
    hsv = np.uint8([[[hue, 220, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_detections(image: np.ndarray, detections: np.ndarray, conf_thr: float = 0.0) -> np.ndarray:
    out = image.copy()
    dets = np.asarray(detections, dtype=np.float32).reshape(-1, 6)
    for det in dets:
        if det[4] < conf_thr:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in det[:4]]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 255), 1)
    return out


def draw_tracks(
    image: np.ndarray,
    views: Sequence[TrackView],
    offset: Tuple[float, float] = (0.0, 0.0),
    annotate_ids: bool = True,
    hud_text: Optional[str] = None,
) -> np.ndarray:
    """Нарисовать треки в экранных координатах (мир + offset)."""
    out = image.copy()
    dx, dy = float(offset[0]), float(offset[1])
    for view in views:
        x1, y1, x2, y2 = view.box
        p1 = (int(round(x1 + dx)), int(round(y1 + dy)))
        p2 = (int(round(x2 + dx)), int(round(y2 + dy)))
        color = _STATE_COLOR.get(view.state, (200, 200, 200))
        thickness = 2 if view.state == TrackState.CONFIRMED else 1
        cv2.rectangle(out, p1, p2, color, thickness)
        if annotate_ids and view.state != TrackState.TENTATIVE:
            label = str(view.track_id)
            if view.count_event:
                label += "+"
            cv2.putText(out, label, (p1[0], max(12, p1[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    if hud_text:
        cv2.rectangle(out, (0, 0), (out.shape[1], 24), (30, 30, 30), -1)
        cv2.putText(out, hud_text, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (240, 240, 240), 1, cv2.LINE_AA)
    return out


def draw_tree_labels(image: np.ndarray, labels: Iterable[Tuple[Tuple[int, int], str]]) -> np.ndarray:
    """labels: [((x, y), text), ...] в экранных координатах."""
    out = image
    for (x, y), text in labels:
        cv2.putText(out, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out


def annotate_frame(
    image: np.ndarray,
    views: Sequence[TrackView],
    offset: Tuple[float, float],
    frame_idx: int,
    counted_total: int,
    annotate_ids: bool = True,
) -> np.ndarray:
    hud = f"frame {frame_idx} | unique apples: {counted_total}"
    return draw_tracks(image, views, offset=offset, annotate_ids=annotate_ids, hud_text=hud)
