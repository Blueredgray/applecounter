"""Абстракция детектора и единый формат результатов.

Формат детекций: np.ndarray (N, 6) float32 [x1, y1, x2, y2, conf, cls].
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..config import Config

EMPTY_DETECTIONS = np.zeros((0, 6), dtype=np.float32)


def detections_array(boxes: np.ndarray, scores: np.ndarray, classes: Optional[np.ndarray] = None) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if classes is None:
        classes = np.zeros(len(boxes), dtype=np.float32)
    else:
        classes = np.asarray(classes, dtype=np.float32).reshape(-1)
    if len(boxes) == 0:
        return EMPTY_DETECTIONS.copy()
    return np.stack([boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], scores, classes], axis=1).astype(np.float32)


class Detector(ABC):
    """Детектор яблок на одном кадре (BGR)."""

    name: str = "base"

    @abstractmethod
    def detect(self, image: np.ndarray) -> np.ndarray:
        """Вернуть массив (N, 6): [x1, y1, x2, y2, conf, cls]."""

    def close(self) -> None:  # pragma: no cover - заглушка
        return None


def build_detector(cfg: Config) -> Detector:
    """Фабрика детекторов по конфигурации."""
    name = str(cfg.get("detector.name", "hsv")).lower()
    if name == "hsv":
        from .color_hsv import ColorHsvDetector

        return ColorHsvDetector(cfg)
    if name == "yolo":
        from .yolo import YoloDetector

        return YoloDetector(cfg)
    raise ValueError(f"неизвестный детектор: {name} (ожидается hsv|yolo)")
