"""Классический детектор яблок по цвету (HSV) — без зависимостей от глубокого обучения.

Подходит как промежуточный/эталонный вариант и для синтетических проверок;
основной вариант для статьи — дообученный YOLO (см. detect.yolo).
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from ..config import Config
from ..utils import geometry as geo
from .base import EMPTY_DETECTIONS, Detector, detections_array


def _kernel(size: int) -> np.ndarray:
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


class ColorHsvDetector(Detector):
    name = "hsv"

    def __init__(self, cfg: Config):
        hsv = cfg.get("detector.hsv", {}) or {}
        self.lower1 = np.array(hsv.get("lower1", [0, 70, 60]), dtype=np.uint8)
        self.upper1 = np.array(hsv.get("upper1", [12, 255, 255]), dtype=np.uint8)
        self.lower2 = np.array(hsv.get("lower2", [168, 70, 60]), dtype=np.uint8)
        self.upper2 = np.array(hsv.get("upper2", [180, 255, 255]), dtype=np.uint8)
        self.close_ksize = int(hsv.get("close_ksize", 3))
        self.open_ksize = int(hsv.get("open_ksize", 1))
        self.min_area = float(hsv.get("min_area", cfg.get("detector.min_area", 24)))
        self.min_circularity = float(hsv.get("min_circularity", 0.55))
        self.nms_iou = float(cfg.get("detector.nms_iou", 0.4))
        self.conf_threshold = float(cfg.get("detector.conf_threshold", 0.3))

    def red_mask(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower1, self.upper1)
        mask2 = cv2.inRange(hsv, self.lower2, self.upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        if self.close_ksize > 1:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel(self.close_ksize))
        if self.open_ksize > 1:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel(self.open_ksize))
        return mask

    def detect(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return EMPTY_DETECTIONS.copy()
        mask = self.red_mask(image)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: List[List[float]] = []
        scores: List[float] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            box_area = float(w * h)
            if box_area < self.min_area:
                continue
            peri = cv2.arcLength(contour, True)
            circularity = 4.0 * np.pi * contour_area(contour) / (peri * peri + 1e-9)
            if circularity < self.min_circularity:
                continue
            # «уверенность»: доля пикселей маски внутри бокса (насыщенность заливки)
            roi = mask[y : y + h, x : x + w]
            fill = float((roi > 0).sum()) / max(box_area, 1.0)
            conf = float(np.clip(0.5 + 0.5 * fill, 0.0, 1.0)) * (0.7 + 0.3 * min(1.0, circularity))
            if conf < self.conf_threshold:
                continue
            boxes.append([x, y, x + w, y + h])
            scores.append(conf)

        if not boxes:
            return EMPTY_DETECTIONS.copy()

        boxes_arr = np.asarray(boxes, dtype=np.float32)
        scores_arr = np.asarray(scores, dtype=np.float32)
        keep = geo.nms(boxes_arr, scores_arr, iou_thr=self.nms_iou)
        return detections_array(boxes_arr[keep], scores_arr[keep])


def contour_area(contour: np.ndarray) -> float:
    return float(cv2.contourArea(contour))
