"""Работа с видео: чтение (с прореживанием/поворотом/масштабом) и запись."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


class VideoReader:
    """Послательный доступ к кадрам видео с опциональными stride/resize/rotate."""

    def __init__(
        self,
        path: str,
        stride: int = 1,
        resize: Optional[Tuple[int, int]] = None,
        rotate90: int = 0,
        max_frames: Optional[int] = None,
    ):
        self.path = str(path)
        if not Path(self.path).exists():
            raise FileNotFoundError(f"видео не найдено: {self.path}")
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise IOError(f"не удалось открыть видео: {self.path}")
        self.stride = max(1, int(stride))
        self.resize = tuple(resize) if resize else None
        self.rotate90 = int(rotate90)
        self.max_frames = max_frames

        self._src_fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._src_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._src_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self._src_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # -- свойства --------------------------------------------------------
    @property
    def source_fps(self) -> float:
        return float(self._src_fps)

    @property
    def source_frame_count(self) -> int:
        return self._src_count

    @property
    def fps(self) -> float:
        return float(self._src_fps) / self.stride

    @property
    def frame_count(self) -> int:
        n = self._src_count // self.stride if self._src_count else 0
        if self.max_frames is not None:
            n = min(n, int(self.max_frames))
        return n

    @property
    def frame_size(self) -> Tuple[int, int]:
        """(width, height) после преобразований."""
        w, h = self._src_w, self._src_h
        if self.rotate90 in (1, -1):
            w, h = h, w
        if self.resize:
            w, h = self.resize
        return int(w), int(h)

    def _transform(self, frame: np.ndarray) -> np.ndarray:
        if self.rotate90 == 1:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotate90 == -1:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if self.resize:
            frame = cv2.resize(frame, self.resize, interpolation=cv2.INTER_AREA)
        return frame

    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        """Итерация (исходный индекс кадра, преобразованный кадр)."""
        index = 0
        emitted = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            if index % self.stride == 0:
                yield index, self._transform(frame)
                emitted += 1
                if self.max_frames is not None and emitted >= int(self.max_frames):
                    break
            index += 1

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class VideoWriter:
    """Запись annotated-видео."""

    def __init__(self, path: str, fps: float, size: Tuple[int, int], fourcc: str = "mp4v"):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(
            self.path, cv2.VideoWriter_fourcc(*fourcc), float(fps), (int(size[0]), int(size[1]))
        )
        if not self._writer.isOpened():
            raise IOError(f"не удалось открыть файл для записи: {self.path}")
        self.size = (int(size[0]), int(size[1]))

    def write(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if (w, h) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_LINEAR)
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_video_frames(path: str, stride: int = 1, max_frames: Optional[int] = None) -> list:
    """Прочитать кадры в память (для маленьких видео/тестов)."""
    frames = []
    with VideoReader(path, stride=stride, max_frames=max_frames) as reader:
        for _, frame in reader.frames():
            frames.append(frame)
    return frames
