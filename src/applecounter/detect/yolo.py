"""Детектор на базе ultralytics YOLOv8 (опциональная зависимость) + обучение."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..config import Config
from ..utils.logging import get_logger
from .base import EMPTY_DETECTIONS, Detector, detections_array

logger = get_logger()


def _require_ultralytics():
    try:
        import ultralytics  # noqa: F401
    except ImportError as exc:  # pragma: no cover - окружение без ultralytics
        raise ImportError(
            "Для детектора YOLO нужно доустановить пакет: pip install ultralytics torch\n"
            "Либо переключиться на классический детектор: --set detector.name=hsv"
        ) from exc


class YoloDetector(Detector):  # pragma: no cover - требует ultralytics
    name = "yolo"

    def __init__(self, cfg: Config):
        _require_ultralytics()
        from ultralytics import YOLO

        ycfg = cfg.get("detector.yolo", {}) or {}
        weights = str(ycfg.get("weights", "yolov8n.pt"))
        path = Path(weights)
        if not path.exists() and not weights.endswith(".pt"):
            weights = weights + ".pt"
        if not Path(weights).exists() and "/" not in weights and "\\" not in weights:
            logger.warning("веса %s не найдены локально — ultralytics попробует загрузить из сети", weights)
        self.model = YOLO(weights)
        self.imgsz = int(ycfg.get("imgsz", 640))
        self.conf = float(ycfg.get("conf", 0.25))
        self.iou = float(ycfg.get("iou", 0.45))
        self.class_id = ycfg.get("class_id", None)
        self.device = str(cfg.get("project.device", "auto"))

    def detect(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return EMPTY_DETECTIONS.copy()
        device = None if self.device == "auto" else self.device
        results = self.model.predict(
            image, imgsz=self.imgsz, conf=self.conf, iou=self.iou, verbose=False, device=device
        )
        if not results:
            return EMPTY_DETECTIONS.copy()
        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return EMPTY_DETECTIONS.copy()
        boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = res.boxes.conf.cpu().numpy().astype(np.float32)
        classes = res.boxes.cls.cpu().numpy().astype(np.float32)
        if self.class_id is not None:
            mask = classes == float(self.class_id)
            boxes, scores, classes = boxes[mask], scores[mask], classes[mask]
        return detections_array(boxes, scores, classes)


def train_yolo(cfg: Config, data_yaml: str, epochs: int = 80, model: str = "yolov8n.pt",
               out_dir: str = "weights", batch: int = 16, imgsz: int = 640,
               name: Optional[str] = None) -> Path:
    """Дообучить YOLO на подготовленном датасете (data.yaml в формате ultralytics)."""
    _require_ultralytics()
    from ultralytics import YOLO

    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"data.yaml не найден: {data_yaml}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    trainer = YOLO(model)
    run_name = name or Path(data_yaml).parent.name
    results = trainer.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str((out / "runs").resolve()),  # абсолютный путь: ultralytics
        name=run_name,                          # иначе уводит в runs/detect/<...>
        seed=int(cfg.get("project.seed", 42)),
        device=None if cfg.get("project.device", "auto") == "auto" else cfg.get("project.device"),
        workers=int(cfg.get("train.workers", 4)),
        resume=bool(cfg.get("train.resume", False)),
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    logger.info("обучение завершено, лучшие веса: %s", best)
    return best
