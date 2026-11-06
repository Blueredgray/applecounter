"""Отчёты и графики для статьи: мозаика пролёта, кривые, столбцы, схема метода."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..config import Config
from ..counting.counter import CountResult
from ..eval.metrics import detection_metrics
from ..trees.segment import TreeMap
from ..utils.video import VideoReader
from ..yieldpred.model import YieldForecast

_TREE_COLORS = plt.get_cmap("tab10")


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Кривая PR
# ---------------------------------------------------------------------------
def pr_curve(preds: Sequence[np.ndarray], gts: Sequence[np.ndarray], iou_thr: float = 0.5):
    """Глобальная кривая PR: сопоставления в порядке убывания уверенности."""
    items = []
    for fi, (pred, gt) in enumerate(zip(preds, gts)):
        pred = np.asarray(pred, dtype=np.float32).reshape(-1, 6)
        gt = np.asarray(gt, dtype=np.float32).reshape(-1, 6)
        for det in pred:
            items.append((float(det[4]), fi, det[:4]))
    n_gt = int(sum(len(np.asarray(g).reshape(-1, 6)) for g in gts))
    if not items or n_gt == 0:
        return np.array([0.0, 1.0]), np.array([1.0, 1.0]), 0.0

    items.sort(key=lambda x: -x[0])
    gt_used = [np.zeros(len(np.asarray(g).reshape(-1, 6)), dtype=bool) for g in gts]
    gt_boxes = [np.asarray(g, dtype=np.float32).reshape(-1, 6)[:, :4] for g in gts]

    from ..utils.geometry import iou_matrix

    tps: List[float] = []
    fps: List[float] = []
    cum_tp = cum_fp = 0
    for _, fi, box in items:
        boxes = gt_boxes[fi]
        is_tp = False
        if len(boxes):
            ious = iou_matrix(np.asarray(box, dtype=np.float32).reshape(1, 4), boxes)[0]
            order = np.argsort(-ious)
            for gi in order:
                if ious[gi] >= iou_thr and not gt_used[fi][gi]:
                    gt_used[fi][gi] = True
                    is_tp = True
                    break
        if is_tp:
            cum_tp += 1
        else:
            cum_fp += 1
        tps.append(cum_tp / n_gt)
        fps.append(cum_fp)

    recall = np.asarray(tps)
    precision = np.asarray([tp / max(tp + fp, 1e-9) for tp, fp in zip(tps, fps)])
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    ap = 0.0
    prev = 0.0
    for r, p in zip(recall, precision):
        ap += (r - prev) * p
        prev = r
    return recall, precision, float(ap)


def fig_pr(preds, gts, path: Path, iou_thr: float = 0.5) -> Path:
    recall, precision, ap = pr_curve(preds, gts, iou_thr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, lw=2, label=f"AP@{iou_thr:.2f} = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Качество детекции яблок")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    return _save(fig, Path(path))


# ---------------------------------------------------------------------------
# Столбчатые диаграммы
# ---------------------------------------------------------------------------
def unit_label(count_unit: str, plural: bool = False) -> str:
    """Человекочитаемая единица счёта без подмены stretch отдельным деревом."""
    labels = {
        "tree": ("дерево", "деревьям"),
        "stretch": ("stretch", "stretch"),
        "sequence": ("последовательность", "последовательностям"),
    }
    one, many = labels.get(str(count_unit), (str(count_unit), str(count_unit)))
    return many if plural else one


def fig_counts_per_tree(count_result: CountResult, gt_counts: Optional[Dict[int, int]], path: Path,
                        count_unit: str = "tree") -> Path:
    keys = sorted(set(count_result.per_tree) | set(gt_counts or {}))
    pred = [count_result.per_tree.get(k, 0) for k in keys]
    fig, ax = plt.subplots(figsize=(max(5, 0.55 * len(keys) + 2), 4))
    x = np.arange(len(keys))
    width = 0.4 if gt_counts else 0.7
    ax.bar(x - width / 2, pred, width, label="расчёт", color="#2b8cbe")
    if gt_counts:
        gt = [int((gt_counts or {}).get(k, 0)) for k in keys]
        ax.bar(x + width / 2, gt, width, label="контроль", color="#f0a35e")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}" for k in keys])
    ax.set_xlabel(f"№ {unit_label(count_unit)}")
    ax.set_ylabel("Количество яблок")
    ax.set_title(f"Подсчёт по {unit_label(count_unit, plural=True)}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, Path(path))


def fig_cumulative(per_frame_ids: Sequence[int], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(np.arange(len(per_frame_ids)), per_frame_ids, lw=2, color="#31a354")
    ax.set_xlabel("№ кадра")
    ax.set_ylabel("Уникальных яблок (суммарно)")
    ax.set_title("Накопленный счёт без повторов")
    ax.grid(alpha=0.3)
    return _save(fig, Path(path))


def fig_yield(forecast: YieldForecast, path: Path, count_unit: str = "tree") -> Path:
    fig, ax = plt.subplots(figsize=(max(5, 0.55 * len(forecast.points) + 2), 4))
    ids = [p.tree_id for p in forecast.points]
    mass = [p.mass_kg for p in forecast.points]
    yerr = [[p.mass_kg - p.ci_lo for p in forecast.points], [p.ci_hi - p.mass_kg for p in forecast.points]]
    x = np.arange(len(ids))
    ax.bar(x, mass, color="#756bb1", yerr=yerr, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in ids])
    ax.set_xlabel(f"№ {unit_label(count_unit)}")
    ax.set_ylabel("Прогноз массы, кг")
    ax.set_title(f"Прогноз урожая (всего {forecast.total_kg:.1f} кг)")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, Path(path))


def fig_error_per_tree(count_rows: Sequence[dict], path: Path, count_unit: str = "tree") -> Path:
    keys = [r["tree_id"] for r in count_rows]
    errs = [r["error"] for r in count_rows]
    colors = ["#31a354" if e == 0 else ("#e6550d" if abs(e) > 3 else "#3182bd") for e in errs]
    fig, ax = plt.subplots(figsize=(max(5, 0.55 * len(keys) + 2), 3.6))
    ax.bar(np.arange(len(keys)), errs, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels([str(k) for k in keys])
    ax.set_xlabel(f"№ {unit_label(count_unit)}")
    ax.set_ylabel("Ошибка счёта")
    ax.set_title(f"Ошибка подсчёта по {unit_label(count_unit, plural=True)} (расчёт − контроль)")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, Path(path))


# ---------------------------------------------------------------------------
# Мозаика пролёта
# ---------------------------------------------------------------------------
def make_mosaic_figure(
    video_path: str,
    offsets: np.ndarray,
    cfg: Config,
    records: Sequence,
    out_path: Path,
    tree_map: Optional[TreeMap] = None,
    stride: int = 4,
    scale: float = 0.5,
) -> Path:
    """Сшить кадры в мировую «мозаику» и отметить на ней найденные яблока/деревья."""
    offsets = np.asarray(offsets, dtype=np.float64).reshape(-1, 2)
    if len(offsets) == 0:
        raise ValueError("нет данных о смещениях камеры")

    with VideoReader(
        video_path,
        stride=int(cfg.get("video.frame_stride", 1)),
        resize=tuple(cfg.get("video.resize")) if cfg.get("video.resize") else None,
        rotate90=int(cfg.get("video.rotate90", 0)),
        max_frames=cfg.get("video.max_frames"),
    ) as reader:
        fw, fh = reader.frame_size
        size = np.array([fw, fh], dtype=np.float64)
        # границы мира: кадр i покрывает [-cum_i, size - cum_i]
        min_w = np.min(-offsets, axis=0)
        max_w = np.max(np.array([size - off for off in offsets]), axis=0)
        span = np.maximum(max_w - min_w, 1.0)
        canvas_w = int(span[0] * scale) + 2
        canvas_h = int(span[1] * scale) + 2
        canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

        processed = 0
        for i, (_, frame) in enumerate(reader.frames()):
            if i >= len(offsets):
                break
            if i % max(1, stride) != 0:
                continue
            small = frame
            if scale != 1.0:
                import cv2

                small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            top_left = ((-offsets[i] - min_w) * scale).astype(int)
            x, y = int(top_left[0]), int(top_left[1])
            h, w = small.shape[:2]
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(canvas_w, x + w), min(canvas_h, y + h)
            if x1 > x0 and y1 > y0:
                canvas[y0:y1, x0:x1] = small[y0 - y : y1 - y, x0 - x : x1 - x]
            processed += 1
        if processed == 0:
            raise ValueError("не удалось собрать мозаику (нет кадров)")

    def to_canvas(world_xy: np.ndarray) -> np.ndarray:
        return (np.asarray(world_xy, dtype=float) - min_w) * scale

    # точки плодов
    for rec in records:
        pt = to_canvas(np.asarray(rec.world_center, dtype=float))
        tree = int(getattr(rec, "tree_id", 0))
        color = tuple(int(c * 255) for c in _TREE_COLORS((tree - 1) % 10)[:3])[::-1]
        import cv2

        cv2.circle(canvas, (int(pt[0]), int(pt[1])), 4, color, -1, lineType=cv2.LINE_AA)

    # границы деревьев
    bnd = np.zeros(0)
    if tree_map is not None:
        if tree_map.method == "axis" and len(tree_map.boundaries):
            bnd = np.asarray(tree_map.boundaries, dtype=float)
        elif len(getattr(tree_map, "sub_edges", np.zeros(0))) >= 3:
            # внутренние границы под-сегментов (subdivide)
            bnd = np.asarray(tree_map.sub_edges, dtype=float)[1:-1]
    if len(bnd):
        import cv2

        for s in bnd:
            q = tree_map.origin + tree_map.direction * s
            normal = np.array([-tree_map.direction[1], tree_map.direction[0]])
            p0 = to_canvas(q + normal * 5000)
            p1 = to_canvas(q - normal * 5000)
            cv2.line(canvas, tuple(p0.astype(int)), tuple(p1.astype(int)), (0, 0, 0), 1, cv2.LINE_AA)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import cv2

    cv2.imwrite(str(out_path), canvas)

    # рядом — график с цветовой легендой деревьев
    fig, ax = plt.subplots(figsize=(6, 3))
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=_TREE_COLORS((t - 1) % 10),
                   markersize=8, label=f"дерево {t}")
        for t in sorted({int(getattr(r, "tree_id", 0)) for r in records} - {0})
    ]
    ax.legend(handles=handles, loc="center")
    ax.axis("off")
    ax.set_title("Цвета точек на мозаике")
    _save(fig, out_path.with_name(out_path.stem + "_legend.png"))
    return out_path


# ---------------------------------------------------------------------------
# Схема метода (для статьи)
# ---------------------------------------------------------------------------
def fig_pipeline(path: Path) -> Path:
    steps = [
        ("Видео пролёта\nвдоль ряда", "#dbe9f6"),
        ("Детекция яблок\n(HSV / YOLO;\nобщий кэш)", "#fff3e0"),
        ("Трекинг без повторов\n(ByteTrack /\nBoT-SORT / DeepSORT)", "#f3e5f5"),
        ("Назначение\nединиц счёта\n(stretch / дерево)", "#e0f7fa"),
        ("MOT- и count-метрики\n(AP, MOTA, IDF1,\nMAE/RMSE)", "#e8f5e9"),
        ("Воспроизводимый\nотчёт и манифест", "#fce4ec"),
    ]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    box_w, box_h = 1.55, 1.5
    x0, y0 = 0.2, 0.9
    for i, (text, color) in enumerate(steps):
        x = x0 + i * (box_w + 0.22)
        ax.add_patch(
            plt.Rectangle((x, y0), box_w, box_h, facecolor=color, edgecolor="#455a64",
                          linewidth=1.2, zorder=2)
        )
        ax.text(x + box_w / 2, y0 + box_h / 2, text, ha="center", va="center",
                fontsize=8.5, zorder=3)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + box_w + 0.2, y0 + box_h / 2),
                        xytext=(x + box_w, y0 + box_h / 2),
                        arrowprops=dict(arrowstyle="-|>", color="#37474f", lw=1.6))
    ax.set_title("Структура пайплайна подсчёта яблок", fontsize=11)
    return _save(fig, Path(path))


# ---------------------------------------------------------------------------
# Отчёт markdown
# ---------------------------------------------------------------------------
def write_report(
    out_dir: Path,
    summary: dict,
    count_result: CountResult,
    det_metrics: Optional[dict] = None,
    count_metrics: Optional[dict] = None,
    track_metrics: Optional[dict] = None,
    forecast: Optional[YieldForecast] = None,
    figures: Optional[Dict[str, str]] = None,
) -> Path:
    out_dir = Path(out_dir)
    count_unit = str(summary.get("count_unit", "tree"))
    one = unit_label(count_unit)
    many = unit_label(count_unit, plural=True)
    lines: List[str] = []
    lines.append("# Отчёт подсчёта яблок\n")
    lines.append(f"- видео: `{summary.get('video', '')}`")
    lines.append(f"- кадров обработано: {summary.get('frames', 0)}")
    lines.append(f"- детектор: {summary.get('detector', '')}")
    if count_unit == "tree":
        lines.append(f"- деревьев на карте: {summary.get('n_trees', 0)}")
    elif count_unit == "stretch":
        lines.append(f"- участков (stretch) на карте: {summary.get('n_trees', 0)}")
    else:
        lines.append(f"- единиц на карте: {summary.get('n_trees', 0)}")
    lines.append(f"- найдено уникальных яблок: **{count_result.total}**")
    if count_result.unassigned:
        lines.append(f"- без привязки к {one}: {count_result.unassigned}")

    if det_metrics:
        lines.append("\n## Детекция\n")
        lines.append("| метрика | значение |")
        lines.append("|---|---|")
        lines.append(f"| AP@{det_metrics['iou_thr']:.2f} | {det_metrics['ap50']:.3f} |")
        lines.append(f"| Precision | {det_metrics['precision']:.3f} |")
        lines.append(f"| Recall | {det_metrics['recall']:.3f} |")
        lines.append(f"| F1 | {det_metrics['f1']:.3f} |")

    if track_metrics:
        lines.append("\n## Трекинг\n")
        lines.append("| метрика | значение |")
        lines.append("|---|---|")
        lines.append(f"| MOTA | {track_metrics['mota']:.3f} |")
        if "idf1" in track_metrics:
            lines.append(f"| IDF1 | {track_metrics['idf1']:.3f} |")
        lines.append(f"| подмены ID (IDSW) | {track_metrics['id_switches']} |")
        lines.append(f"| FP / FN | {track_metrics['fp']} / {track_metrics['fn']} |")

    if count_metrics:
        lines.append(f"\n## Точность счёта по {many}\n")
        lines.append("| метрика | значение |")
        lines.append("|---|---|")
        lines.append(f"| итог расчёт / контроль | {count_metrics['pred_total']} / {count_metrics['gt_total']} |")
        lines.append(f"| ошибка итога, % | {count_metrics['total_error_pct']:.1f} |")
        lines.append(f"| MAE (на единицу) | {count_metrics['mae']:.2f} |")
        lines.append(f"| RMSE | {count_metrics['rmse']:.2f} |")
        lines.append(f"| MAPE, % | {count_metrics['mape']:.1f} |")
        lines.append(f"| смещение (bias) | {count_metrics['bias']:.2f} |")
        lines.append(f"| единиц в допуске ±{count_metrics['tolerance']} | {count_metrics['within_tolerance']}/{count_metrics['n_trees']} |")

    lines.append(f"\n## Счёт по {many}\n")
    lines.append(f"| {one} | найдено |")
    lines.append("|---|---|")
    for tree_id, cnt in count_result.per_tree.items():
        lines.append(f"| {tree_id} | {cnt} |")
    if 0 in count_result.per_tree:
        lines.append(f"\n_{one} 0 — объекты вне карты единиц счёта (просветы/фон)_")

    if forecast:
        lines.append(f"\n## Прогноз урожая (метод: {forecast.method}; тип: {forecast.estimate_type})\n")
        lines.append(f"- всего: **{forecast.total_kg:.1f} кг** "
                     f"(95% ДИ: {forecast.ci_lo:.1f} … {forecast.ci_hi:.1f} кг)")
        if forecast.params:
            lines.append(f"- параметры: `{json.dumps(forecast.params, ensure_ascii=False)}`")
        lines.append(f"\n| {one} | яблок | масса, кг | 95% ДИ |")
        lines.append("|---|---|---|---|")
        for p in forecast.points:
            lines.append(f"| {p.tree_id} | {p.count} | {p.mass_kg:.1f} | {p.ci_lo:.1f} … {p.ci_hi:.1f} |")

    if figures:
        lines.append("\n## Графики\n")
        for title, rel in figures.items():
            lines.append(f"### {title}\n")
            lines.append(f"![{title}]({rel})\n")

    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
