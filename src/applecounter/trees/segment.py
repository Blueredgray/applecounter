"""Разбиение пролёта на деревья.

Поддерживаются три способа:
  * axis       — автоматический: мировые координаты детекций проецируются на ось
                 пролёта (PCA), плотность точек сглаживается, «просветы» между
                 кронами дают границы деревьев;
  * segments   — сегменты по кадрам (формат VideoFruitCounting: файл SFRAME/csv);
  * boundaries — ручные границы в мировых координатах (JSON).
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import Config

_TREE_KEYS = re.compile(r"tree|дерев", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Модель
# ---------------------------------------------------------------------------
@dataclass
class TreeMap:
    """Привязка объектов к деревьям."""

    method: str = "axis"
    origin: np.ndarray = field(default_factory=lambda: np.zeros(2))
    direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    boundaries: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # для метода по кадрам: список (f_start, f_end, tree_id)
    frame_segments: List[Tuple[int, int, int]] = field(default_factory=list)
    # мировые границы под-сегментов (subdivide): [e0 < e1 < ... < eM],
    # дерево k (1..M) покрывает отрезок (e_{k-1}, e_k]
    sub_edges: np.ndarray = field(default_factory=lambda: np.zeros(0))
    meta: dict = field(default_factory=dict)

    @property
    def n_trees(self) -> int:
        if self.method in ("segments", "frames"):
            if self.frame_segments:
                return len({seg[2] for seg in self.frame_segments})
            return 0
        return len(self.boundaries) + 1

    def tree_ids(self) -> List[int]:
        if self.method in ("segments", "frames"):
            return sorted({seg[2] for seg in self.frame_segments})
        return list(range(1, self.n_trees + 1))

    def project(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        return (pts - self.origin) @ self.direction

    def assign_points(self, points: np.ndarray) -> np.ndarray:
        """Мировые координаты -> id деревьев (0 — вне сегментов/просветов)."""
        if self.method in ("segments", "frames"):
            raise ValueError("для метода segments используйте assign_frames")
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if len(pts) == 0:
            return np.zeros(0, dtype=int)
        if len(self.boundaries) == 0:
            return np.ones(len(pts), dtype=int)
        s = self.project(pts)
        ids = np.searchsorted(self.boundaries, s) + 1
        return ids.astype(int)

    def assign_frames(self, frame_idx: int) -> int:
        """Кадр -> id деревья (0 — вне сегментов)."""
        for f0, f1, tree in self.frame_segments:
            if f0 <= frame_idx <= f1:
                return int(tree)
        return 0

    def assign_track(self, world_positions: np.ndarray, frames: Sequence[int]) -> int:
        """Трек (мировые точки + кадры) -> id деревья."""
        if self.method in ("segments", "frames"):
            # при посегментном разбиении (subdivide) назначение идёт по мировым
            # координатам — точнее медианы кадра (наблюдение яблока асимметрично)
            if len(self.sub_edges) >= 2 and world_positions is not None and len(world_positions):
                ids = self._assign_by_sub_edges(world_positions)
                values, counts = np.unique(ids, return_counts=True)
                best = int(values[np.argmax(counts)])
                return best if best > 0 else 0
            if not len(frames):
                return 0
            return self.assign_frames(int(np.median(np.asarray(frames, dtype=int))))
        if len(world_positions) == 0:
            return 0
        ids = self.assign_points(world_positions)
        ids = ids[ids > 0]
        if len(ids) == 0:
            return 0
        # большинство голосов
        values, counts = np.unique(ids, return_counts=True)
        return int(values[np.argmax(counts)])

    def _assign_by_sub_edges(self, points: np.ndarray) -> np.ndarray:
        """Мировые точки -> id под-сегментов (0 — вне ряда)."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        s = (pts - self.origin) @ self.direction
        # дерево k покрывает (e_{k-1}, e_k]: side='left' считает границы строго меньше s
        ids = np.searchsorted(self.sub_edges, s, side="left")
        m = len(self.sub_edges) - 1
        ids = np.where((ids >= 1) & (ids <= m), ids, 0)
        return ids.astype(int)


# ---------------------------------------------------------------------------
# Построение
# ---------------------------------------------------------------------------
def _axis_tree_map(points: np.ndarray, cfg_trees: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Вернуть (origin, direction, boundaries, meta) по облаку мировых точек."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 4:
        return np.zeros(2), np.array([1.0, 0.0]), np.zeros(0), {"reason": "too_few_points"}

    origin = pts.mean(axis=0)
    centered = pts - origin
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    norm = np.linalg.norm(direction)
    direction = direction / (norm if norm > 1e-9 else 1.0)

    s = centered @ direction
    lo, hi = float(s.min()), float(s.max())
    span = hi - lo
    if span <= 1e-6:
        return origin, direction, np.zeros(0), {"reason": "zero_span"}

    bin_width = 4.0
    bins = max(8, int(np.ceil(span / bin_width)) + 1)
    counts, edges = np.histogram(s, bins=bins, range=(lo, hi))
    centers = 0.5 * (edges[1:] + edges[0:-1])

    sigma = float(cfg_trees.get("smooth_sigma", 7.0))
    sigma_bins = max(0.5, sigma / max(bin_width, 1e-6))
    # гауссово сглаживание вручную (без scipy.ndimage)
    radius = int(np.ceil(3 * sigma_bins))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= kernel.sum()
    smooth = np.convolve(np.pad(counts.astype(float), radius, mode="edge"), kernel, mode="valid")

    peak = float(smooth.max())
    if peak <= 0:
        return origin, direction, np.zeros(0), {"reason": "empty_density"}
    threshold = float(cfg_trees.get("gap_ratio", 0.20)) * peak

    above = smooth >= threshold
    # интервалы «крон»
    intervals: List[Tuple[int, int]] = []
    start = None
    for i, flag in enumerate(above):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            intervals.append((start, i - 1))
            start = None
    if start is not None:
        intervals.append((start, len(above) - 1))

    min_points = int(cfg_trees.get("min_points", 12))
    min_span = float(cfg_trees.get("min_span", 40))
    kept: List[Tuple[int, int]] = []
    for a, b in intervals:
        width = float(edges[b + 1] - edges[a])
        n_pts = int(((s >= edges[a]) & (s <= edges[b + 1])).sum())
        if width >= min_span and n_pts >= min_points:
            kept.append((a, b))

    if not kept:
        # запасной вариант: делим диапазон на равные части (если задано n_trees)
        n_trees = int(cfg_trees.get("n_trees") or 0)
        if n_trees >= 1:
            boundaries = np.linspace(lo, hi, n_trees + 1)[1:-1]
            return origin, direction, boundaries, {"reason": "equal_split", "n_trees": n_trees}
        return origin, direction, np.zeros(0), {"reason": "no_intervals"}

    boundaries = []
    for (a1, b1), (a2, b2) in zip(kept, kept[1:]):
        gap_lo, gap_hi = edges[b1 + 1], edges[a2]
        boundaries.append(0.5 * (gap_lo + gap_hi))

    n_fixed = int(cfg_trees.get("n_trees") or 0)
    if n_fixed and len(boundaries) + 1 != n_fixed and n_fixed >= 1:
        # если просветов не хватает/слишком много — равномерно по диапазону точек
        boundaries = list(np.linspace(lo, hi, n_fixed + 1)[1:-1])

    meta = {
        "n_intervals": len(kept),
        "density_peak": peak,
        "threshold": float(threshold),
        "span": float(span),
        "s_min": float(lo),
        "s_max": float(hi),
        "centers": [float(c) for c in centers[:: max(1, len(centers) // 20)]],
        "density": [float(c) for c in smooth[:: max(1, len(smooth) // 20)]],
    }
    return origin, direction, np.asarray(boundaries, dtype=float), meta


def build_tree_map(points: np.ndarray, cfg: Config, video_name: Optional[str] = None,
                   detections: Optional[np.ndarray] = None) -> TreeMap:
    """Собрать TreeMap по конфигурации (метод auto выбирает подходящий).

    detections — необязательные детекции Nx5 [x1,y1,x2,y2,кадр] в мировых
    координатах: нужны для subdivide (оценка масштаба px/м по размеру боксов
    яблок и положение сегментов вдоль оси ряда).
    """
    trees_cfg = cfg.get("trees", {}) or {}
    method = str(trees_cfg.get("method", "auto")).lower()

    if method == "auto":
        if trees_cfg.get("segments_file"):
            method = "segments"
        elif trees_cfg.get("boundaries_file"):
            method = "boundaries"
        elif trees_cfg.get("n_trees"):
            method = "fixed"
        else:
            method = "axis"

    if method in ("segments", "frames"):
        seg_file = trees_cfg.get("segments_file")
        segments = load_segments(str(seg_file), video_name=video_name) if seg_file else []
        meta: dict = {"file": seg_file}
        origin = np.zeros(2)
        direction = np.array([1.0, 0.0])
        sub_edges = np.zeros(0)
        if bool(trees_cfg.get("subdivide", True)) and segments \
                and detections is not None and len(detections) >= 10:
            try:
                segments, origin, direction, sub_edges, sub_meta = _subdivide_segments(
                    segments, detections, trees_cfg)
                meta.update(sub_meta)
            except Exception as exc:  # не ломаем прогон из-за разбиения
                meta["subdivide_error"] = str(exc)
        return TreeMap(method="segments", frame_segments=segments, origin=origin,
                       direction=direction, sub_edges=sub_edges, meta=meta)

    if method == "boundaries":
        origin, direction, _, _ = _axis_tree_map(points, trees_cfg)
        data = json.loads(Path(str(trees_cfg["boundaries_file"])).read_text(encoding="utf-8"))
        if isinstance(data, dict) and "axis" in data:
            # {"axis": [[x0, y0], [x1, y1]], "boundaries": [..]}
            a0, a1 = np.asarray(data["axis"], dtype=float)
            origin = np.asarray(a0, dtype=float)
            vec = np.asarray(a1, dtype=float) - origin
            direction = vec / (np.linalg.norm(vec) or 1.0)
            boundaries = np.asarray(data["boundaries"], dtype=float)
        elif isinstance(data, dict) and data.get("axis") == "x":
            origin = np.zeros(2)
            direction = np.array([1.0, 0.0])
            boundaries = np.asarray(data["boundaries"], dtype=float)
        else:
            # простой список границ по оси X
            origin = np.zeros(2)
            direction = np.array([1.0, 0.0])
            boundaries = np.asarray(data if isinstance(data, list) else data.get("boundaries", []), dtype=float)
        boundaries = np.sort(boundaries)
        return TreeMap(method="axis", origin=origin, direction=direction, boundaries=boundaries,
                       meta={"source": "boundaries_file"})

    # axis / fixed
    cfg_trees = dict(trees_cfg)
    origin, direction, boundaries, meta = _axis_tree_map(points, cfg_trees)
    return TreeMap(method="axis", origin=origin, direction=direction, boundaries=boundaries, meta=meta)


def _subdivide_segments(
    segments: List[Tuple[int, int, int]],
    detections: np.ndarray,
    trees_cfg: dict,
) -> Tuple[List[Tuple[int, int, int]], np.ndarray, np.ndarray, np.ndarray, dict]:
    """Разбить каждый сегмент (stretch) на отдельные деревья.

    Масштаб (px/м) оценивается по среднему эквивалентному диаметру боксов
    детекций: физический диаметр яблока принят за ``apple_diameter_mm``.
    Положение границы сегмента вдоль ряда берётся по сглаженному медианному
    профилю детекций кадра (мировая ось — PCA детекций).

    Возвращает (сегменты с новыми tree_id, origin, direction, sub_edges, meta).
    """
    det = np.asarray(detections, dtype=np.float64)
    if det.ndim != 2 or det.shape[1] < 5:
        raise ValueError("для subdivide нужны детекции Nx5 [x1,y1,x2,y2,кадр]")
    if not segments:
        raise ValueError("нет сегментов для разбиения")

    x1, y1, x2, y2 = det[:, 0], det[:, 1], det[:, 2], det[:, 3]
    frames_all = np.rint(det[:, 4]).astype(int)
    diam = np.sqrt(np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0))
    diam = diam[diam > 2.0]
    med_d = float(np.median(diam)) if diam.size else 0.0

    apple_m = max(float(trees_cfg.get("apple_diameter_mm", 75.0)), 1.0) / 1000.0
    px_per_m = med_d / apple_m if med_d > 0 else 0.0
    spacing_m = max(float(trees_cfg.get("spacing_m", 1.2)), 0.1)
    spacing_px = spacing_m * px_per_m if px_per_m > 0 else 0.0
    fixed_n = int(trees_cfg.get("trees_per_stretch", 0) or 0)

    # ось ряда (PCA по центрам боксов)
    centers = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5], axis=1)
    origin = centers.mean(axis=0)
    _, _, vt = np.linalg.svd(centers - origin, full_matrices=False)
    direction = vt[0]
    nrm = float(np.linalg.norm(direction))
    direction = direction / (nrm if nrm > 1e-12 else 1.0)
    s_all = (centers - origin) @ direction

    # знак оси: s обязан расти по ходу движения камеры, иначе профиль по
    # кадрам убывает и монотонизация превращает его в константу
    q_lo, q_hi = np.quantile(frames_all, [0.1, 0.9])
    if float(np.median(s_all[frames_all <= q_lo])) > float(np.median(s_all[frames_all >= q_hi])):
        direction = -direction
        s_all = -s_all

    # профиль положения вдоль ряда по кадрам: медиана детекций кадра,
    # сглаживание и монотонизация (камера едет вдоль ряда в одну сторону)
    f_min = max(0, int(min(frames_all.min(), min(f0 for f0, _, _ in segments))))
    f_max = int(max(frames_all.max(), max(f1 for _, f1, _ in segments)))
    order = np.argsort(frames_all, kind="stable")
    f_sorted, s_sorted = frames_all[order], s_all[order]
    uniq = np.unique(f_sorted)
    lo = np.searchsorted(f_sorted, uniq, side="left")
    hi = np.searchsorted(f_sorted, uniq, side="right")
    med = np.array([float(np.median(s_sorted[a:b])) for a, b in zip(lo, hi)], dtype=np.float64)

    win = int(trees_cfg.get("subdivide_smooth", 31) or 31)
    if win % 2 == 0:
        win += 1
    if win > 1 and med.size >= win:
        r = win // 2
        med = np.convolve(np.pad(med, r, mode="edge"), np.ones(win) / win, mode="valid")

    frame_axis = np.arange(f_min, f_max + 1)
    s_of = np.interp(frame_axis, uniq.astype(float), med)
    s_of = np.maximum.accumulate(s_of)

    new_segments: List[Tuple[int, int, int]] = []
    edges: List[float] = []
    tree_id = 0
    n_no_scale = 0
    for idx, (f0, f1, _sid) in enumerate(segments):
        fa = min(max(int(f0), f_min), f_max)
        fb = min(max(int(f1), f_min), f_max)
        s_lo = float(s_of[fa - f_min])
        s_hi = float(s_of[fb - f_min])
        if s_hi < s_lo:
            s_lo, s_hi = s_hi, s_lo
        if fixed_n >= 1:
            n = fixed_n
        elif spacing_px > 0:
            n = max(1, int(round((s_hi - s_lo) / spacing_px)))
        else:
            n = 1
            n_no_scale += 1
        seg_edges = np.linspace(s_lo, s_hi, n + 1)
        if idx == 0:
            edges.append(float(seg_edges[0]))
        for k in range(n):
            tree_id += 1
            a, b = float(seg_edges[k]), float(seg_edges[k + 1])
            edges.append(b)
            # обратный отсчёт: кадр, в котором профиль достигает границы
            fa_k = f_min + int(np.searchsorted(s_of, a, side="left"))
            fb_k = f_min + int(np.searchsorted(s_of, b, side="left"))
            fa_k = min(max(fa_k, f0), f1)
            fb_k = min(max(fb_k, f0), f1)
            if fb_k < fa_k:
                fb_k = fa_k
            new_segments.append((int(fa_k), int(fb_k), tree_id))

    sub_edges = np.maximum.accumulate(np.asarray(edges, dtype=np.float64))
    meta = {
        "subdivided": True,
        "px_per_m": round(px_per_m, 2),
        "spacing_px": round(spacing_px, 2),
        "spacing_m": spacing_m,
        "apple_diameter_mm": float(trees_cfg.get("apple_diameter_mm", 75.0)),
        "median_apple_px": round(med_d, 2),
        "n_stretches": len(segments),
        "n_trees": tree_id,
        "mean_trees_per_stretch": round(tree_id / max(1, len(segments)), 2),
    }
    if n_no_scale:
        meta["no_scale_segments"] = n_no_scale
    return new_segments, origin, direction, sub_edges, meta


def load_segments(path: str, video_name: Optional[str] = None) -> List[Tuple[int, int, int]]:
    """Загрузить сегменты по кадрам.
    Поддерживаемые форматы:
      * CSV VideoFruitCounting: <video>,f0,f1,...,fN   (границы stretch'ей)
      * CSV: frame_start,frame_end,tree_id

    Если в файле несколько строк (по строке на видео) — можно передать video_name
    (имя файла без расширения), тогда будет взята соответствующая строка.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"файл сегментов не найден: {path}")

    rows = []
    with file_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.reader(fh):
            if not row:
                continue
            rows.append([cell.strip() for cell in row])

    # отделить строки с именем видео от чистых csv
    named_rows = [r for r in rows if r and not _is_number(r[0])]
    if named_rows:
        if video_name is not None:
            stem = Path(video_name).stem
            chosen = [r for r in named_rows if Path(r[0]).stem == stem]
            if not chosen:
                raise ValueError(
                    f"в {path} нет строки для видео {video_name!r}; доступны: "
                    + ", ".join(r[0] for r in named_rows[:5])
                )
            rows = chosen
        elif len(named_rows) == 1:
            rows = named_rows
        else:
            raise ValueError(
                f"в {path} несколько видео — укажите имя видео (встречаются: "
                + ", ".join(r[0] for r in named_rows[:5]) + ")"
            )

    segments: List[Tuple[int, int, int]] = []
    for row in rows:
        numeric_start = 0
        if not _is_number(row[0]):
            numeric_start = 1
        nums = []
        for cell in row[numeric_start:]:
            try:
                nums.append(int(float(cell)))
            except ValueError:
                break
        if len(nums) >= 3 and _strictly_increasing(nums):
            # границы сегментов: t0 < t1 < ... < tN -> N отрезков с id 1..N
            # (VideoFruitCounting: video,f0,f1,...,fN)
            for i in range(len(nums) - 1):
                if nums[i + 1] > nums[i]:
                    segments.append((nums[i], nums[i + 1], i + 1))
        elif len(nums) >= 3:
            # f0, f1, tree_id
            segments.append((nums[0], nums[1], int(nums[2])))
        elif len(nums) == 2:
            # единственный отрезок без id дерева
            segments.append((nums[0], nums[1], 1))

    segments.sort(key=lambda item: (item[2], item[0]))
    if not segments:
        raise ValueError(f"не удалось разобрать сегменты из {path}")
    return segments


def _strictly_increasing(values: Sequence[int]) -> bool:
    return all(b > a for a, b in zip(values, values[1:]))


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False
