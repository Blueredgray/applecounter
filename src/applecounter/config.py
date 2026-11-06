"""Конфигурация пайплайна: YAML + встроенные значения по умолчанию + CLI-оверрайды."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import yaml

DEFAULTS: dict = {
    "project": {"seed": 42, "device": "auto"},
    "paths": {
        "data_root": "data",
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "synthetic_dir": "data/synthetic",
        "runs_dir": "outputs",
    },
    "video": {
        "max_frames": None,
        "frame_stride": 1,
        "resize": None,
        "rotate90": 0,
        "fps": None,
    },
    "detector": {
        "name": "hsv",
        "conf_threshold": 0.30,
        "nms_iou": 0.40,
        "min_area": 24,
        "hsv": {
            "lower1": [0, 70, 60],
            "upper1": [12, 255, 255],
            "lower2": [168, 70, 60],
            "upper2": [180, 255, 255],
            "close_ksize": 3,
            "open_ksize": 1,
            "min_area": 24,
            "min_circularity": 0.55,
        },
        "yolo": {
            "weights": "weights/apples_yolov8n.pt",
            "imgsz": 640,
            "conf": 0.25,
            "iou": 0.45,
            "class_id": 0,
        },
    },
    "stabilizer": {
        "enabled": True,
        "method": "orb",
        "max_features": 1500,
        "reproj_thresh": 3.0,
        "min_matches": 12,
    },
    "tracker": {
        "name": "ours",
        "high_conf": 0.45,
        "low_conf": 0.15,
        "min_iou": 0.25,
        "max_age": 25,
        "reid_max_age": 60,
        "n_init": 2,
        "use_appearance": True,
        "appearance_gate": 0.55,
        "reid_max_dist": 60.0,
        "velocity_damping": 0.7,
    },
    "trees": {
        "method": "auto",
        "gap_ratio": 0.20,
        "min_points": 12,
        "min_span": 40,
        "smooth_sigma": 7.0,
        "n_trees": None,
        "segments_file": None,
        "boundaries_file": None,
        # посегментное разбиение (метод segments) на отдельные деревья:
        # каждый сегмент (stretch) делится на n = длина/шаг равных отрезков
        "subdivide": True,
        # шаг и диаметр подобраны под сад датасета VFC (IRTA Mollerussa,
        # Story Inored, fruiting wall): 1.0 м в ряду x 3.6 м междурядье;
        # stretch = 5 м = 5 деревьев; опорный диаметр 70 мм даёт масштаб,
        # при котором средняя длина stretch'а (~2095 px) = 5.0 м
        "spacing_m": 1.0,            # шаг между деревьями вдоль ряда, м
        "apple_diameter_mm": 70.0,   # опорный диаметр яблока для оценки px/м
        "trees_per_stretch": 0,      # >0 — фиксированное число деревьев в сегменте
    },
    "counting": {"min_obs": 2, "merge_dist": 14.0},
    "yield": {
        "method": "physical",
        "mean_mass_g": 180.0,
        "visibility_factor": 1.0,
        "visibility_sigma": 0.08,
        "mass_sigma_g": 25.0,
        "regression_csv": None,
        "ridge_lambda": 1.0e-3,
        "confidence": 0.95,
    },
    "eval": {"iou_match": 0.50, "count_tolerance": 3},
    "output": {
        "save_video": True,
        "save_mosaic": True,
        "save_figures": True,
        "annotate_ids": True,
        "mosaic_stride": 4,
        "mosaic_scale": 0.5,
    },
}


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """Рекурсивное слияние словарей (override выигрывает)."""
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce(value: Any) -> Any:
    """Разбор значения CLI-оверрайда через YAML-семантику."""
    if not isinstance(value, str):
        return value
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


class Config:
    """Словарь конфигурации с доступом по точечному пути: cfg.get('tracker.n_init')."""

    def __init__(self, data: Optional[Mapping[str, Any]] = None):
        self.data: dict = copy.deepcopy(DEFAULTS) if data is None else copy.deepcopy(dict(data))

    # -- доступ ---------------------------------------------------------
    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in path.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, path: str, value: Any) -> None:
        parts = path.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise KeyError(f"нельзя задать {path}: {part} не словарь")
        node[parts[-1]] = value

    def apply_overrides(self, overrides: Iterable[str]) -> None:
        """Оверрайды вида 'tracker.n_init=3'."""
        for item in overrides or ():
            if "=" not in item:
                raise ValueError(f"оверрайд должен быть key=value, получено: {item!r}")
            key, _, raw = item.partition("=")
            self.set(key.strip(), _coerce(raw.strip()))

    def __getitem__(self, path: str) -> Any:
        value = self.get(path)
        if value is None:
            raise KeyError(path)
        return value

    def __repr__(self) -> str:  # pragma: no cover
        return f"Config({self.data})"


def load_config(path: Optional[str] = None, overrides: Optional[Iterable[str]] = None) -> Config:
    """Загрузить конфигурацию: дефолты <- YAML-файл <- CLI-оверрайды."""
    data = copy.deepcopy(DEFAULTS)
    if path:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"файл конфигурации не найден: {file_path}")
        with file_path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"конфигурация должна быть словарём: {file_path}")
        data = deep_merge(data, loaded)
    cfg = Config(data)
    if overrides:
        cfg.apply_overrides(overrides)
    return cfg
