"""Внешний вид объекта (цветовая гистограмма) для re-identification треков."""

from __future__ import annotations

import cv2
import numpy as np

from ..utils import geometry as geo


def box_histogram(image: np.ndarray, box, bins=(8, 8, 4)) -> np.ndarray:
    """Нормированная HSV-гистограмма внутри бокса [x1, y1, x2, y2]."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    x1, x2 = int(max(0, min(w - 1, x1))), int(max(0, min(w, x2)))
    y1, y2 = int(max(0, min(h - 1, y1))), int(max(0, min(h, y2)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return np.zeros(int(np.prod(bins)), dtype=np.float32)

    roi = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(bins), [0, 180, 0, 256, 0, 256])
    hist = hist.astype(np.float32).ravel()
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist /= norm
    return hist


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def appearance_cost(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """Стоимость для назначения: 1 - косинусное сходство."""
    return 1.0 - cosine(hist_a, hist_b)


def extract_for_dets(image: np.ndarray, world_boxes: np.ndarray, offset) -> list:
    """Гистограммы для детекций: мир. боксы сдвигаются в экранные координаты."""
    if image is None or len(world_boxes) == 0:
        return []
    screen_boxes = geo.shift(world_boxes, float(offset[0]), float(offset[1]))
    return [box_histogram(image, box) for box in screen_boxes]
