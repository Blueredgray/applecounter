"""Компенсация движения камеры (стабилизация) — переход в «мировые» координаты.

Для пролёта вдоль ряда принимаем модель чистого сдвига: между кадрами оцениваем
аффинное преобразование (ORB + RANSAC) и накапливаем только вектор переноса.
Мировые координаты: world = screen - cum_offset.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..config import Config


class Stabilizer:
    """Потоковая оценка накопленного смещения камеры."""

    def __init__(self, cfg: Config):
        s = cfg.get("stabilizer", {}) or {}
        self.enabled = bool(s.get("enabled", True))
        self.method = str(s.get("method", "orb")).lower()
        self.max_features = int(s.get("max_features", 1500))
        self.reproj_thresh = float(s.get("reproj_thresh", 3.0))
        self.min_matches = int(s.get("min_matches", 12))

        self._orb = cv2.ORB_create(nfeatures=self.max_features) if self.method == "orb" else None
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING) if self.method == "orb" else None
        self._prev_gray: Optional[np.ndarray] = None
        self._cum = np.zeros(2, dtype=np.float64)
        self._last_t = np.zeros(2, dtype=np.float64)
        self.offsets: List[np.ndarray] = []
        self.n_estimated = 0
        self.n_failed = 0

    def reset(self) -> None:
        self._prev_gray = None
        self._cum = np.zeros(2, dtype=np.float64)
        self._last_t = np.zeros(2, dtype=np.float64)
        self.offsets = []
        self.n_estimated = 0
        self.n_failed = 0

    # -- шаг ------------------------------------------------------------
    def step(self, gray: np.ndarray) -> np.ndarray:
        """Обработать кадр, вернуть накопленное смещение (2,) для этого кадра."""
        if not self.enabled or self.method == "none":
            self.offsets.append(self._cum.copy())
            return self._cum.copy()

        t = np.zeros(2, dtype=np.float64)
        if self._prev_gray is not None:
            t = self._estimate(self._prev_gray, gray)
            if t is None:
                self.n_failed += 1
                t = self._last_t.copy()  # держим прежний сдвиг
            else:
                self.n_estimated += 1
                self._last_t = t
            self._cum = self._cum + t

        self._prev_gray = gray
        self.offsets.append(self._cum.copy())
        return self._cum.copy()

    def _estimate(self, prev: np.ndarray, cur: np.ndarray) -> Optional[np.ndarray]:
        pts1, des1 = self._orb.detectAndCompute(prev, None)
        pts2, des2 = self._orb.detectAndCompute(cur, None)
        if des1 is None or des2 is None or len(des1) < self.min_matches or len(des2) < self.min_matches:
            return None
        matches = self._matcher.knnMatch(des1, des2, k=2)
        good = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)
        if len(good) < self.min_matches:
            return None
        src = np.float32([pts1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([pts2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, _ = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=self.reproj_thresh
        )
        if matrix is None:
            return None
        return np.array([matrix[0, 2], matrix[1, 2]], dtype=np.float64)

    # -- преобразования -------------------------------------------------
    def to_world(self, frame_index: int, xy) -> np.ndarray:
        """Экранные координаты -> мировые."""
        offset = self.offsets[frame_index] if 0 <= frame_index < len(self.offsets) else self._cum
        arr = np.asarray(xy, dtype=np.float64)
        return arr - offset

    def to_screen(self, frame_index: int, xy) -> np.ndarray:
        offset = self.offsets[frame_index] if 0 <= frame_index < len(self.offsets) else self._cum
        arr = np.asarray(xy, dtype=np.float64)
        return arr + offset

    @property
    def path_length(self) -> float:
        if len(self.offsets) < 2:
            return 0.0
        diffs = np.diff(np.stack(self.offsets, axis=0), axis=0)
        return float(np.linalg.norm(diffs, axis=1).sum())
