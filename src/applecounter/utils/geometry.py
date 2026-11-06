"""Геометрия боксов: IoU, NMS, конвертации."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def to_array(boxes: Sequence) -> np.ndarray:
    """Привести набор боксов к массиву (N, 4) float32 [x1, y1, x2, y2]."""
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return arr.reshape(-1, 4)


def area(boxes: np.ndarray) -> np.ndarray:
    boxes = to_array(boxes)
    return np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Матрица IoU (N, M) для двух наборов боксов."""
    a = to_array(a)
    b = to_array(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area(a)[:, None] + area(b)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def iou_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU попарно для двух массивов одинаковой длины (N,)."""
    a = to_array(a)
    b = to_array(b)
    if len(a) != len(b):
        raise ValueError("для iou_pairs нужны массивы одной длины")
    if len(a) == 0:
        return np.zeros(0, dtype=np.float32)
    lt = np.maximum(a[:, :2], b[:, :2])
    rb = np.minimum(a[:, 2:], b[:, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[:, 0] * wh[:, 1]
    union = area(a) + area(b) - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.4) -> List[int]:
    """Классический NMS: возвращает индексы оставленных боксов."""
    boxes = to_array(boxes)
    scores = np.asarray(scores, dtype=np.float32)
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = iou_matrix(boxes[i : i + 1], boxes[rest])[0]
        order = rest[ious <= iou_thr]
    return keep


def clip_boxes(boxes: np.ndarray, width: float, height: float) -> np.ndarray:
    boxes = to_array(boxes).copy()
    if len(boxes) == 0:
        return boxes
    boxes[:, 0] = np.clip(boxes[:, 0], 0, width)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, width)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, height)
    return boxes


def center(boxes: np.ndarray) -> np.ndarray:
    boxes = to_array(boxes)
    if len(boxes) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.stack([(boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5], axis=1)


def shift(boxes: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Сдвинуть боксы (для перевода мир. <-> экранные координаты).

    Принимает (N, 4) и (N, 6) [x1, y1, x2, y2, ...]: сдвигаются только
    координаты, остальные колонки (conf, cls) сохраняются.
    """
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1) if arr.size else arr.reshape(0, 4)
    if arr.size == 0:
        width = arr.shape[1] if arr.ndim == 2 and arr.shape[1] else 4
        return np.zeros((0, width), dtype=np.float32)
    arr = arr.copy()
    arr[:, 0] += dx
    arr[:, 2] += dx
    arr[:, 1] += dy
    arr[:, 3] += dy
    return arr


def xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    boxes = to_array(boxes)
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return np.stack([boxes[:, 0] + w * 0.5, boxes[:, 1] + h * 0.5, w, h], axis=1)


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    half_w = boxes[:, 2] * 0.5
    half_h = boxes[:, 3] * 0.5
    return np.stack([boxes[:, 0] - half_w, boxes[:, 1] - half_h, boxes[:, 0] + half_w, boxes[:, 1] + half_h], axis=1)
