"""Синтетический «пролёт вдоль яблочного ряда» с полным ground truth.

Зачем нужно: видео с пролётом дадут позже, поэтому для отладки и воспроизводимых
экспериментов генерируется собственный ролик, где известны:
  * ровно одинаковое (истинное) число яблок на каждом дереве;
  * положение каждого яблока в мировых координатах и номере дерева;
  * боксы видимых яблок в каждом кадре (для оценки детекции и трекинга).

Модель сцены: ряд деревьев в мировых координатах, камера едет вдоль ряда
(со сглаженным дрожанием и лёгким изменением высоты), часть яблок закрыта
листьями, которые колышутся — видимость таких плодов меняется от кадра к кадру.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class SyntheticParams:
    width: int = 640
    height: int = 360
    fps: float = 25.0
    n_trees: int = 6
    apples_min: int = 18
    apples_max: int = 30
    spacing: float = 280.0        # расстояние между деревьями, мир. px
    speed: float = 130.0          # скорость камеры, мир. px/с
    radius_min: float = 7.0
    radius_max: float = 11.0
    front_ratio: float = 0.6      # доля «передних» (не закрытых) яблок
    occluder_ratio: float = 1.3   # радиус листа относительно яблока
    sway_period_s: Tuple[float, float] = (3.0, 6.5)
    seed: int = 42
    distractors: int = 0          # «ложные» красные пятна на кроне
    noise_sigma: float = 3.0
    blur: bool = True
    frames: Optional[int] = None  # явное число кадров (иначе считаем из скорости)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["sway_period_s"] = list(self.sway_period_s)
        return data


@dataclass
class Apple:
    apple_id: int
    tree_id: int
    x: float
    y: float
    r: float
    layer: int            # 0 — передний (виден всегда), 1 — задний (может быть закрыт)
    hue_shift: int
    phase: float = 0.0
    base_d: float = 0.0
    amp: float = 0.0
    period: float = 4.0
    color: Tuple[int, int, int] = (40, 60, 215)


def _circle_overlap(r: float, R: float, d: float) -> float:
    """Площадь пересечения двух окружностей (формула линзы)."""
    if d >= r + R:
        return 0.0
    if d <= abs(R - r):
        return math.pi * min(r, R) ** 2
    try:
        a = r * r * math.acos((d * d + r * r - R * R) / (2 * d * r))
        b = R * R * math.acos((d * d + R * R - r * r) / (2 * d * R))
        c = 0.5 * math.sqrt(max((-d + r + R) * (d + r - R) * (d - r + R) * (d + r + R), 0.0))
    except ValueError:
        return 0.0
    return a + b - c


class SyntheticOrchard:
    """Генератор сцены и кадров."""

    def __init__(self, params: Optional[SyntheticParams] = None):
        self.p = params or SyntheticParams()
        self.rng = np.random.default_rng(self.p.seed)
        self.apples: List[Apple] = []
        self.leaves: List[dict] = []       # статичные «листья» (мировые координаты)
        self.trunks: List[dict] = []
        self.canopies: List[dict] = []
        self.distractors: List[dict] = []
        self._build_scene()

    # -- сцена ------------------------------------------------------------
    def _build_scene(self) -> None:
        p = self.p
        cy = 0.0                      # центр кроны в мировых координатах (ось Y)
        rx = min(p.spacing * 0.40, p.width * 0.20)
        ry = p.height * 0.33

        apple_id = 1
        for t in range(p.n_trees):
            cx = t * p.spacing
            self.canopies.append({"x": cx, "y": cy, "rx": rx, "ry": ry, "tree": t + 1})
            self.trunks.append({"x": cx, "y": cy + ry * 0.75, "w": max(8.0, rx * 0.14)})

            # листва (статичные пятна внутри кроны)
            for _ in range(int(self.rng.integers(26, 40))):
                ang = self.rng.uniform(0, 2 * math.pi)
                rad = math.sqrt(self.rng.uniform(0, 1)) * 0.92
                lx = cx + math.cos(ang) * rad * rx
                ly = cy + math.sin(ang) * rad * ry
                self.leaves.append(
                    {
                        "x": lx,
                        "y": ly,
                        "rx": float(self.rng.uniform(rx * 0.10, rx * 0.22)),
                        "ry": float(self.rng.uniform(ry * 0.10, ry * 0.22)),
                        "angle": float(self.rng.uniform(0, 180)),
                        "shade": float(self.rng.uniform(0.55, 1.25)),
                        "tree": t + 1,
                    }
                )

            n_apples = int(self.rng.integers(p.apples_min, p.apples_max + 1))
            placed = 0
            attempts = 0
            while placed < n_apples and attempts < n_apples * 60:
                attempts += 1
                r = float(self.rng.uniform(p.radius_min, p.radius_max))
                ang = self.rng.uniform(0, 2 * math.pi)
                rad = math.sqrt(self.rng.uniform(0, 1)) * 0.82
                ax = cx + math.cos(ang) * rad * rx
                ay = cy + math.sin(ang) * rad * ry
                # не даём яблокам соседних деревьев наложиться
                if any(abs(ax - a.x) < (r + a.r) * 1.1 and abs(ay - a.y) < (r + a.r) * 1.1
                       for a in self.apples if a.tree_id == t + 1):
                    continue
                layer = 0 if self.rng.random() < p.front_ratio else 1
                period = float(self.rng.uniform(*p.sway_period_s))
                base_d = float(self.rng.uniform(0.7 * r, 2.3 * r))
                amp = float(self.rng.uniform(0.2 * r, 1.5 * r))
                # вариант цвета: от насыщенного красного до тёмно-бордового/желтоватого
                pick = self.rng.random()
                if pick < 0.7:
                    color = (35, 60, 215)
                elif pick < 0.9:
                    color = (30, 90, 235)
                else:
                    color = (60, 90, 190)
                self.apples.append(
                    Apple(
                        apple_id=apple_id,
                        tree_id=t + 1,
                        x=float(ax),
                        y=float(ay),
                        r=float(r),
                        layer=int(layer),
                        hue_shift=int(self.rng.integers(-4, 5)),
                        phase=float(self.rng.uniform(0, 2 * math.pi)),
                        base_d=base_d,
                        amp=amp,
                        period=period,
                        color=color,
                    )
                )
                apple_id += 1
                placed += 1

        for _ in range(int(p.distractors)):
            t = int(self.rng.integers(0, p.n_trees))
            canon = self.canopies[t]
            self.distractors.append(
                {
                    "x": float(canon["x"] + self.rng.uniform(-0.8, 0.8) * canon["rx"]),
                    "y": float(canon["y"] + self.rng.uniform(-0.8, 0.8) * canon["ry"]),
                    "r": float(self.rng.uniform(p.radius_min * 0.7, p.radius_max * 0.9)),
                    "color": (70, 110, 170),
                }
            )

    # -- геометрия камеры -------------------------------------------------
    def camera_at(self, frame: int) -> Tuple[float, float, float]:
        p = self.p
        t_sec = frame / p.fps
        cam_x = -p.width * 0.62 + p.speed * t_sec
        cam_x += 5.0 * math.sin(2 * math.pi * t_sec * 0.17)          # продольное дрожание
        cam_y = 3.0 * math.sin(2 * math.pi * t_sec * 0.23 + 1.0)     # вертикальное дрожание
        scale = 1.0 + 0.015 * math.sin(2 * math.pi * t_sec * 0.11)   # лёгкое «дыхание» высоты
        return cam_x, cam_y, scale

    def project(self, x: float, y: float, frame: int) -> Tuple[float, float]:
        cam_x, cam_y, scale = self.camera_at(frame)
        sx = (x - cam_x) * scale + self.p.width * 0.5
        sy = (y - cam_y) * scale + self.p.height * 0.46
        return sx, sy

    def n_frames(self) -> int:
        p = self.p
        if p.frames:
            return int(p.frames)
        distance = p.spacing * max(0, p.n_trees - 1) + p.width * 1.24
        return int(math.ceil(distance * p.fps / p.speed)) + 5

    # -- видимость --------------------------------------------------------
    def apple_visible(self, apple: Apple, frame: int) -> Tuple[bool, float]:
        if apple.layer == 0:
            return True, 1.0
        d = apple.base_d + apple.amp * math.sin(2 * math.pi * frame / (apple.period * self.p.fps) + apple.phase)
        d = max(0.0, d)
        R = self.p.occluder_ratio * apple.r
        overlap = _circle_overlap(apple.r, R, d)
        frac = 1.0 - overlap / (math.pi * apple.r ** 2)
        return frac >= 0.6, max(0.0, min(1.0, frac))

    def _ensure_visibility(self, n_frames: int) -> None:
        """Каждое заднее яблоко должно быть видно хотя бы в одном кадре."""
        step = max(1, n_frames // 40)
        for apple in self.apples:
            if apple.layer == 0:
                continue
            seen = any(self.apple_visible(apple, f)[0] for f in range(0, n_frames, step))
            if not seen:
                apple.base_d = 2.6 * apple.r   # уводим лист в сторону
                apple.amp = 0.3 * apple.r

    # -- отрисовка --------------------------------------------------------
    def _draw_apple(self, canvas: np.ndarray, sx: float, sy: float, r: float,
                    color: Tuple[int, int, int], hue_shift: int = 0) -> None:
        x, y, rad = int(round(sx)), int(round(sy)), max(2, int(round(r)))
        h, w = canvas.shape[:2]
        x0, x1 = max(0, x - rad), min(w, x + rad + 1)
        y0, y1 = max(0, y - rad), min(h, y + rad + 1)
        if x1 <= x0 or y1 <= y0:
            return
        sub = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.float32)
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        mask = dist <= rad
        if not mask.any():
            return
        base = np.array(color, dtype=np.float32)
        if hue_shift:
            base = np.clip(base + np.array([hue_shift, 0, -hue_shift], dtype=np.float32), 0, 255)
        shade = np.clip(1.05 - 0.5 * (dist / max(rad, 1e-6)) ** 1.5, 0.45, 1.05)
        for c in range(3):
            sub[..., c] = base[c] * shade
        # блик
        hx, hy = x - rad * 0.35, y - rad * 0.35
        hdist = np.sqrt((xx - hx) ** 2 + (yy - hy) ** 2)
        highlight = np.clip(1.0 - hdist / max(rad * 0.5, 1e-6), 0, 1) ** 2
        for c in range(3):
            sub[..., c] = np.clip(sub[..., c] + highlight * 90, 0, 255)
        region = canvas[y0:y1, x0:x1]
        region[mask] = sub[mask].astype(np.uint8)

    def render_frame(self, frame: int) -> np.ndarray:
        p = self.p
        # небо и земля
        canvas = np.zeros((p.height, p.width, 3), dtype=np.uint8)
        sky_top = np.array([235, 205, 175], dtype=np.float32)
        sky_bottom = np.array([245, 225, 205], dtype=np.float32)
        horizon = int(p.height * 0.72)
        for y in range(horizon):
            a = y / max(1, horizon)
            canvas[y, :] = (sky_top + (sky_bottom - sky_top) * a).astype(np.uint8)
        ground_top = np.array([120, 165, 120], dtype=np.float32)
        ground_bottom = np.array([70, 120, 85], dtype=np.float32)
        for y in range(horizon, p.height):
            a = (y - horizon) / max(1, p.height - horizon)
            canvas[y, :] = (ground_top + (ground_bottom - ground_top) * a).astype(np.uint8)

        # кроны и стволы
        for canopy in self.canopies:
            cx, cy = self.project(canopy["x"], canopy["y"], frame)
            rx = canopy["rx"]
            ry = canopy["ry"]
            cv2.ellipse(canvas, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360,
                        (70, 130, 60), -1, lineType=cv2.LINE_AA)
        for trunk in self.trunks:
            tx, ty = self.project(trunk["x"], trunk["y"], frame)
            cv2.rectangle(canvas, (int(tx - trunk["w"] / 2), int(ty)),
                          (int(tx + trunk["w"] / 2), p.height), (55, 70, 110), -1)
        for leaf in self.leaves:
            lx, ly = self.project(leaf["x"], leaf["y"], frame)
            shade = leaf["shade"]
            color = (int(60 * shade), int(135 * shade), int(55 * shade))
            color = tuple(int(np.clip(c, 0, 255)) for c in color)
            cv2.ellipse(canvas, (int(lx), int(ly)),
                        (max(2, int(leaf["rx"])), max(2, int(leaf["ry"]))),
                        leaf["angle"], 0, 360, color, -1, lineType=cv2.LINE_AA)

        # задние яблоки
        back: List[Tuple[Apple, float, float]] = []
        for apple in self.apples:
            if apple.layer != 1:
                continue
            sx, sy = self.project(apple.x, apple.y, frame)
            if sx < -30 or sx > p.width + 30 or sy < -30 or sy > p.height + 30:
                continue
            self._draw_apple(canvas, sx, sy, apple.r, apple.color, apple.hue_shift)
            back.append((apple, sx, sy))

        # «листья», закрывающие задние яблоки (колышутся)
        for apple, sx, sy in back:
            d = apple.base_d + apple.amp * math.sin(
                2 * math.pi * frame / (apple.period * p.fps) + apple.phase
            )
            d = max(0.0, d)
            ang = apple.phase + frame * 0.01
            ox = sx + d * math.cos(ang)
            oy = sy + d * math.sin(ang)
            R = p.occluder_ratio * apple.r
            cv2.circle(canvas, (int(ox), int(oy)), int(R), (65, 140, 55), -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (int(ox - R * 0.3), int(oy - R * 0.3)), int(R * 0.45),
                       (80, 160, 70), -1, lineType=cv2.LINE_AA)

        # передние яблоки
        for apple in self.apples:
            if apple.layer != 0:
                continue
            sx, sy = self.project(apple.x, apple.y, frame)
            if sx < -30 or sx > p.width + 30 or sy < -30 or sy > p.height + 30:
                continue
            self._draw_apple(canvas, sx, sy, apple.r, apple.color, apple.hue_shift)

        # ложные «красные пятна» (опционально)
        for dis in self.distractors:
            sx, sy = self.project(dis["x"], dis["y"], frame)
            if 0 <= sx < p.width and 0 <= sy < p.height:
                self._draw_apple(canvas, sx, sy, dis["r"], dis["color"])

        # освещение/шум/размытие
        t = frame / p.fps
        gain = 1.0 + 0.07 * math.sin(2 * math.pi * t * 0.13)
        canvas = np.clip(canvas.astype(np.float32) * gain, 0, 255)
        if p.noise_sigma > 0:
            canvas = canvas + self.rng.normal(0, p.noise_sigma, canvas.shape)
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
        if p.blur:
            canvas = cv2.GaussianBlur(canvas, (3, 3), 0.6)
        return canvas

    # -- ground truth -----------------------------------------------------
    def per_frame_iter(self, frame: int):
        p = self.p
        rows: List[dict] = []
        for apple in self.apples:
            visible, frac = self.apple_visible(apple, frame)
            if not visible:
                continue
            sx, sy = self.project(apple.x, apple.y, frame)
            x1, y1, x2, y2 = sx - apple.r, sy - apple.r, sx + apple.r, sy + apple.r
            # учитываем только яблоки, целиком (по крайней мере на 60%) в кадре
            cx1, cx2 = max(x1, 0.0), min(x2, float(p.width))
            cy1, cy2 = max(y1, 0.0), min(y2, float(p.height))
            if (cx2 - cx1) < apple.r * 1.2 or (cy2 - cy1) < apple.r * 1.2:
                continue
            if (cx2 - cx1) * (cy2 - cy1) < 0.6 * (apple.r * 2) ** 2:
                continue
            yield_dict = {
                "apple_id": apple.apple_id,
                "tree_id": apple.tree_id,
                "x1": round(float(x1), 2),
                "y1": round(float(y1), 2),
                "x2": round(float(x2), 2),
                "y2": round(float(y2), 2),
                "visible_frac": round(float(frac), 3),
            }
            yield yield_dict


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------
def generate_synthetic(out_dir: str, params: Optional[SyntheticParams] = None,
                       video_name: str = "synthetic_orchard.mp4") -> dict:
    """Сгенерировать видео + ground truth в out_dir; вернуть манифест."""
    params = params or SyntheticParams()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gt_dir = out / "gt"
    gt_dir.mkdir(exist_ok=True)

    gen = SyntheticOrchard(params)
    n_frames = gen.n_frames()
    gen._ensure_visibility(n_frames)

    video_path = out / video_name
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), params.fps,
        (params.width, params.height),
    )
    if not writer.isOpened():
        raise IOError(f"не удалось открыть видео для записи: {video_path}")

    per_frame_rows: List[str] = ["frame,apple_id,tree_id,x1,y1,x2,y2,visible_frac"]
    for frame in range(n_frames):
        image = gen.render_frame(frame)
        writer.write(image)
        for row in gen.per_frame_iter(frame):
            per_frame_rows.append(
                f"{frame},{row['apple_id']},{row['tree_id']},"
                f"{row['x1']},{row['y1']},{row['x2']},{row['y2']},{row['visible_frac']}"
            )
    writer.release()

    # GT: объекты и счёт по деревьям
    objects = [
        {
            "apple_id": a.apple_id,
            "tree_id": a.tree_id,
            "x_world": round(a.x, 2),
            "y_world": round(a.y, 2),
            "radius": round(a.r, 2),
            "layer": a.layer,
        }
        for a in gen.apples
    ]
    with (gt_dir / "objects.csv").open("w", encoding="utf-8") as fh:
        fh.write("apple_id,tree_id,x_world,y_world,radius,layer\n")
        for obj in objects:
            fh.write("{apple_id},{tree_id},{x_world},{y_world},{radius},{layer}\n".format(**obj))

    counts: Dict[int, int] = {}
    for a in gen.apples:
        counts[a.tree_id] = counts.get(a.tree_id, 0) + 1
    with (gt_dir / "gt_counts.csv").open("w", encoding="utf-8") as fh:
        fh.write("tree_id,count\n")
        for tree_id in sorted(counts):
            fh.write(f"{tree_id},{counts[tree_id]}\n")

    (gt_dir / "per_frame.csv").write_text("\n".join(per_frame_rows) + "\n", encoding="utf-8")

    # сегменты по кадрам (для метода segments): когда дерево в кадре
    with (gt_dir / "segments.csv").open("w", encoding="utf-8") as fh:
        fh.write("video,f0,f1,tree_id\n")
        for canopy in gen.canopies:
            f0, f1 = None, None
            for frame in range(n_frames):
                sx, _ = gen.project(canopy["x"], canopy["y"], frame)
                margin = canopy["rx"] * 0.9
                inside = (-margin) <= (sx - params.width * 0.5) <= (params.width + margin)
                if inside:
                    if f0 is None:
                        f0 = frame
                    f1 = frame
            if f0 is not None:
                fh.write(f"{Path(video_name).stem},{f0},{f1},{canopy['tree']}\n")

    manifest = {
        "video": str(video_path),
        "params": params.to_dict(),
        "n_frames": n_frames,
        "n_apples": len(gen.apples),
        "n_trees": params.n_trees,
        "counts": {str(k): v for k, v in sorted(counts.items())},
        "gt": {
            "counts": str(gt_dir / "gt_counts.csv"),
            "objects": str(gt_dir / "objects.csv"),
            "per_frame": str(gt_dir / "per_frame.csv"),
            "segments": str(gt_dir / "segments.csv"),
        },
        "trees_world_x": [c["x"] for c in gen.canopies],
    }
    (gt_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
