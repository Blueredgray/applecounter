"""Сквозной пайплайн: видео -> детекция -> трекинг -> деревья -> счёт -> урожай -> отчёт."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from .config import Config
from .counting.counter import AppleCounter, CountResult
from .detect.base import build_detector
from .eval.metrics import counting_metrics, detection_metrics, tracking_metrics
from .track import build_tracker
from .track.tracker import TrackState
from .trees.segment import TreeMap, build_tree_map
from .trees.stabilize import Stabilizer
from .utils import geometry as geo
from .utils.logging import get_logger
from .utils.video import VideoReader, VideoWriter
from .viz import draw as viz_draw
from .viz import report as viz_report
from .yieldpred.model import YieldForecast, forecast_counts

logger = get_logger()


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
@dataclass
class GroundTruth:
    counts: Dict[int, int] = field(default_factory=dict)
    per_frame: Dict[int, np.ndarray] = field(default_factory=dict)      # src frame -> (N,6)
    tracks: Dict[int, Dict[int, np.ndarray]] = field(default_factory=dict)  # src frame -> {id: box}
    objects: Optional[pd.DataFrame] = None
    source: str = ""

    @property
    def available(self) -> bool:
        return bool(self.counts)


def load_ground_truth(gt_dir: str) -> GroundTruth:
    """Загрузить ground truth (каталог gt/ с gt_counts.csv и per_frame.csv)."""
    root = Path(gt_dir)
    gt = GroundTruth(source=str(root))
    counts_path = root / "gt_counts.csv"
    if counts_path.exists():
        df = pd.read_csv(counts_path)
        key = "tree_id" if "tree_id" in df.columns else df.columns[0]
        val = "count" if "count" in df.columns else df.columns[1]
        gt.counts = {int(r[key]): int(r[val]) for _, r in df.iterrows()}

    per_frame_path = root / "per_frame.csv"
    if per_frame_path.exists():
        df = pd.read_csv(per_frame_path)
        for frame, group in df.groupby("frame"):
            boxes = []
            tracks: Dict[int, np.ndarray] = {}
            for _, row in group.iterrows():
                box = np.array([row["x1"], row["y1"], row["x2"], row["y2"], 1.0, 0.0], dtype=np.float32)
                boxes.append(box)
                tracks[int(row["apple_id"])] = box[:4].astype(np.float32)
            gt.per_frame[int(frame)] = np.asarray(boxes, dtype=np.float32).reshape(-1, 6)
            gt.tracks[int(frame)] = tracks

    objects_path = root / "objects.csv"
    if objects_path.exists():
        gt.objects = pd.read_csv(objects_path)
    return gt


# ---------------------------------------------------------------------------
# Основной прогон
# ---------------------------------------------------------------------------
def run_video(
    cfg: Config,
    video_path: str,
    out_dir: str,
    gt_dir: Optional[str] = None,
) -> dict:
    """Полный проход по видео; возвращает словарь summary (также пишется в summary.json)."""
    t_start = time.time()

    stride = int(cfg.get("video.frame_stride", 1))
    resize = cfg.get("video.resize")
    rotate90 = int(cfg.get("video.rotate90", 0))
    max_frames = cfg.get("video.max_frames")

    reader = VideoReader(
        video_path,
        stride=stride,
        resize=tuple(resize) if resize else None,
        rotate90=rotate90,
        max_frames=max_frames,
    )
    # каталог создаём только после успешного открытия видео,
    # иначе неудачный запуск оставляет пустую папку в outputs/
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figures_dir = out / "figures"
    detector = build_detector(cfg)
    stabilizer = Stabilizer(cfg)
    tracker = build_tracker(cfg)
    counter = AppleCounter(min_obs=int(cfg.get("counting.min_obs", 2)))

    save_video = bool(cfg.get("output.save_video", True))
    annotate_ids = bool(cfg.get("output.annotate_ids", True))
    writer: Optional[VideoWriter] = None

    offsets: List[np.ndarray] = []
    world_points: List[np.ndarray] = []
    preds_by_frame: Dict[int, np.ndarray] = {}
    pred_tracks_by_frame: Dict[int, Dict[int, np.ndarray]] = {}
    frame_rows: List[dict] = []
    tracks_frame_rows: List[dict] = []   # пофреймовые боксы треков (экспорт VFC)
    counted_history: List[int] = []

    gt: Optional[GroundTruth] = GroundTruth()
    if gt_dir:
        gt = load_ground_truth(gt_dir)
        if not gt.available:
            logger.warning("в %s нет gt_counts.csv — оценка точности счёта недоступна", gt_dir)

    n_processed = 0
    for src_idx, frame in reader.frames():
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cum = stabilizer.step(gray)

        dets_screen = detector.detect(frame)
        dets_world = geo.shift(dets_screen, -float(cum[0]), -float(cum[1]))
        if len(dets_world):
            # боксы в мировых координатах + номер кадра (для subdivide карты деревьев)
            world_points.append(np.column_stack(
                [dets_world[:, :4].astype(np.float64),
                 np.full(len(dets_world), float(n_processed))]))

        views = tracker.update(n_processed, dets_world, image=frame, offset=(float(cum[0]), float(cum[1])))
        counter.update(n_processed, views)

        preds_by_frame[src_idx] = dets_screen
        # для оценки трекинга — только подтверждённые треки, реально
        # сопоставленные с детекцией в этом кадре (без «предсказаний» LOST)
        screen_tracks: Dict[int, np.ndarray] = {}
        for v in views:
            if v.state != TrackState.CONFIRMED or not v.updated:
                continue
            box_screen = geo.shift(np.asarray(v.box, dtype=np.float32).reshape(1, 4),
                                   float(cum[0]), float(cum[1]))[0]
            screen_tracks[v.track_id] = box_screen
            tracks_frame_rows.append(
                {
                    "frame": src_idx,
                    "track_id": v.track_id,
                    "x1": float(box_screen[0]),
                    "y1": float(box_screen[1]),
                    "x2": float(box_screen[2]),
                    "y2": float(box_screen[3]),
                    "conf": float(v.conf),
                }
            )
        pred_tracks_by_frame[src_idx] = screen_tracks
        counted_history.append(len(counter._counted_ids))
        frame_rows.append(
            {
                "frame": src_idx,
                "unique_counted": counted_history[-1],
                "active_tracks": int(sum(1 for v in views if v.state != TrackState.TENTATIVE)),
                "detections": int(len(dets_screen)),
                "cam_dx": float(cum[0]),
                "cam_dy": float(cum[1]),
            }
        )
        offsets.append(np.asarray(cum, dtype=np.float64))

        if save_video:
            if writer is None:
                writer = VideoWriter(str(out / "annotated.mp4"), reader.fps, reader.frame_size)
            hud = f"frame {src_idx} | unique {counted_history[-1]}"
            annotated = viz_draw.draw_tracks(frame, views, offset=(float(cum[0]), float(cum[1])),
                                             annotate_ids=annotate_ids, hud_text=hud)
            writer.write(annotated)

        n_processed += 1
        if n_processed % 100 == 0:
            logger.info("обработано кадров: %d | уникальных яблок: %d", n_processed, len(counter._counted_ids))

    if writer is not None:
        writer.close()
    reader.close()
    detector.close()

    logger.info("обработано всего кадров: %d (%.1f c)", n_processed, time.time() - t_start)

    # -- карта деревьев ---------------------------------------------------
    points_arr = np.concatenate(world_points, axis=0) if world_points else np.zeros((0, 5))
    tree_map = build_tree_map(points_arr[:, :2] if len(points_arr) else points_arr, cfg,
                              video_name=Path(video_path).name,
                              detections=points_arr if len(points_arr) else None)
    counts: CountResult = counter.finalize(
        tree_map, tracker.tracks, merge_dist=float(cfg.get("counting.merge_dist", 14.0))
    )
    logger.info("карта деревьев: %d; найдено яблок: %d (без привязки: %d)",
                tree_map.n_trees, counts.total, counts.unassigned)

    # -- прогноз урожая ---------------------------------------------------
    yield_input = {k: v for k, v in counts.per_tree.items() if k > 0}
    forecast: Optional[YieldForecast] = None
    try:
        forecast = forecast_counts(yield_input, cfg)
    except Exception as exc:
        logger.warning("прогноз урожая не рассчитан: %s", exc)

    # -- оценка -----------------------------------------------------------
    det_metrics = track_metrics = count_metrics = None
    if gt is not None and (gt.per_frame or gt.counts):
        common = sorted(set(preds_by_frame) & set(gt.per_frame))
        if common:
            det_metrics = detection_metrics(
                [preds_by_frame[f] for f in common],
                [gt.per_frame[f] for f in common],
                iou_thr=float(cfg.get("eval.iou_match", 0.5)),
            )
            track_metrics = tracking_metrics(
                [pred_tracks_by_frame.get(f, {}) for f in common],
                [gt.tracks[f] for f in common],
                iou_thr=float(cfg.get("eval.iou_match", 0.5)),
            )
        if gt.counts:
            count_metrics = counting_metrics(
                {k: v for k, v in counts.per_tree.items() if k > 0},
                gt.counts,
                tolerance=int(cfg.get("eval.count_tolerance", 3)),
            )

    # -- сохранение -------------------------------------------------------
    summary = {
        "video": str(video_path),
        "frames": n_processed,
        "detector": str(cfg.get("detector.name", "hsv")),
        "detector_params": {
            "name": str(cfg.get("detector.name", "hsv")),
            "weights": cfg.get("detector.yolo.weights") if str(cfg.get("detector.name", "hsv")) == "yolo" else None,
            "imgsz": cfg.get("detector.yolo.imgsz") if str(cfg.get("detector.name", "hsv")) == "yolo" else None,
            "conf": float(cfg.get("detector.conf_threshold", 0.3)),
        },
        "stabilizer": {
            "enabled": bool(cfg.get("stabilizer.enabled", True)),
            "estimated": stabilizer.n_estimated,
            "failed": stabilizer.n_failed,
            "path_length_px": round(stabilizer.path_length, 1),
        },
        "tracker": {
            "name": str(cfg.get("tracker.name", "ours")),
            "unique_counted": counts.total,
            "created_tracks": tracker.n_created,
            "n_double_count_events": counter.n_double_count_events,
        },
        "n_trees": tree_map.n_trees,
        "tree_method": tree_map.method,
        "tree_meta": {**tree_map.meta,
                      "boundaries": [float(b) for b in np.atleast_1d(tree_map.boundaries)],
                      "origin": [float(v) for v in tree_map.origin],
                      "direction": [float(v) for v in tree_map.direction]},
        "counts": {str(k): v for k, v in counts.per_tree.items()},
        "unassigned": counts.unassigned,
        "total_apples": counts.total,
        "yield": None if forecast is None else {
            "method": forecast.method,
            "total_kg": round(forecast.total_kg, 2),
            "sigma_kg": round(forecast.total_sigma, 2),
            "ci_lo": round(forecast.ci_lo, 2),
            "ci_hi": round(forecast.ci_hi, 2),
            "params": forecast.params,
        },
        "detection_metrics": det_metrics,
        "tracking_metrics": track_metrics,
        "count_metrics": count_metrics,
        "elapsed_sec": round(time.time() - t_start, 2),
    }

    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                                      encoding="utf-8")
    pd.DataFrame(counts.as_rows()).to_csv(out / "tracks.csv", index=False)
    if tracks_frame_rows:
        pd.DataFrame(tracks_frame_rows).to_csv(out / "tracks_frame.csv", index=False)
    pd.DataFrame(frame_rows).to_csv(out / "per_frame.csv", index=False)
    pd.DataFrame(
        [{"tree_id": k, "count": v} for k, v in counts.per_tree.items()]
    ).to_csv(out / "counts.csv", index=False)
    if forecast is not None:
        pd.DataFrame(forecast.as_rows()).to_csv(out / "yield.csv", index=False)
        (out / "yield.json").write_text(
            json.dumps(
                {
                    "method": forecast.method,
                    "total_kg": forecast.total_kg,
                    "total_sigma": forecast.total_sigma,
                    "ci": [forecast.ci_lo, forecast.ci_hi],
                    "params": forecast.params,
                    "per_tree": forecast.as_rows(),
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    # -- графики и мозаика -------------------------------------------------
    figures: Dict[str, str] = {}
    if bool(cfg.get("output.save_figures", True)):
        figures_dir.mkdir(exist_ok=True)
        figures["Подсчёт по деревьям"] = viz_report.fig_counts_per_tree(
            counts, gt.counts if gt and gt.counts else None, figures_dir / "counts_per_tree.png"
        ).name
        figures["Накопленный счёт"] = viz_report.fig_cumulative(
            counts.per_frame_ids, figures_dir / "cumulative_count.png"
        ).name
        if det_metrics is not None:
            figures["Кривая PR"] = viz_report.fig_pr(
                [preds_by_frame[f] for f in sorted(set(preds_by_frame) & set(gt.per_frame))],
                [gt.per_frame[f] for f in sorted(set(preds_by_frame) & set(gt.per_frame))],
                figures_dir / "pr_curve.png",
                iou_thr=float(cfg.get("eval.iou_match", 0.5)),
            ).name
        if count_metrics is not None:
            figures["Ошибка по деревьям"] = viz_report.fig_error_per_tree(
                count_metrics["per_tree"], figures_dir / "error_per_tree.png"
            ).name
        if forecast is not None and forecast.points:
            figures["Прогноз урожая"] = viz_report.fig_yield(
                forecast, figures_dir / "yield_forecast.png"
            ).name
        figures["Структура метода"] = viz_report.fig_pipeline(figures_dir / "pipeline.png").name

    if bool(cfg.get("output.save_mosaic", True)) and offsets:
        try:
            mosaic = viz_report.make_mosaic_figure(
                video_path,
                np.stack(offsets, axis=0),
                cfg,
                counts.records,
                out / "mosaic.png",
                tree_map=tree_map,
                stride=int(cfg.get("output.mosaic_stride", 4)),
                scale=float(cfg.get("output.mosaic_scale", 0.5)),
            )
            figures["Мозаика пролёта"] = mosaic.name
        except Exception as exc:
            logger.warning("мозаика не собрана: %s", exc)

    report_path = viz_report.write_report(
        out, summary, counts, det_metrics, count_metrics, track_metrics, forecast,
        figures={k: f"figures/{v}" for k, v in figures.items()},
    )

    # -- артефакты для статьи ---------------------------------------------
    write_latex_assets(out, summary, count_metrics, det_metrics, forecast)

    summary["paths"] = {
        "report": str(report_path),
        "summary": str(out / "summary.json"),
        "annotated": str(out / "annotated.mp4") if (out / "annotated.mp4").exists() else None,
        "figures": {k: str(out / "figures" / v) for k, v in figures.items()},
    }
    return summary


# ---------------------------------------------------------------------------
# LaTeX-артефакты для статьи
# ---------------------------------------------------------------------------
def write_latex_assets(out: Path, summary: dict, count_metrics: Optional[dict],
                       det_metrics: Optional[dict], forecast: Optional[YieldForecast]) -> None:
    """Сформировать results_table.tex и results_long.tex для подстановки в статью."""
    lines = ["% сгенерировано applecounter — не редактировать вручную",
             r"\begin{tabular}{lcc}", r"\hline",
             "Показатель & Значение \\\\", r"\hline"]
    if det_metrics:
        lines += [
            f"AP@{det_metrics['iou_thr']:.2f} & {det_metrics['ap50']:.3f} \\\\",
            f"Precision / Recall & {det_metrics['precision']:.3f} / {det_metrics['recall']:.3f} \\\\",
            f"F1 & {det_metrics['f1']:.3f} \\\\",
        ]
    if summary.get("tracking_metrics") or summary.get("tracking_metrics") is None:
        tm = summary.get("tracking_metrics")
        if tm:
            lines += [
                f"MOTA & {tm['mota']:.3f} \\\\",
                f"Подмены ID & {tm['id_switches']} \\\\",
            ]
    if count_metrics:
        lines += [
            f"Ошибка итогового счёта, \\% & {count_metrics['total_error_pct']:.1f} \\\\",
            f"MAE (на дерево) & {count_metrics['mae']:.2f} \\\\",
            f"RMSE & {count_metrics['rmse']:.2f} \\\\",
            f"MAPE, \\% & {count_metrics['mape']:.1f} \\\\",
            f"Деревьев в допуске & {count_metrics['within_tolerance']}/{count_metrics['n_trees']} \\\\",
        ]
    lines += [r"\hline", r"\end{tabular}"]
    (out / "results_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if forecast is not None:
        rows = ["% сгенерировано applecounter",
                r"\begin{tabular}{lrrr}", r"\hline",
                "Дерево & Яблок & Масса, кг & 95\\% ДИ, кг \\\\", r"\hline"]
        for p in forecast.points:
            rows.append(f"{p.tree_id} & {p.count} & {p.mass_kg:.1f} & {p.ci_lo:.1f}--{p.ci_hi:.1f} \\\\")
        rows += [r"\hline",
                 f"Итого & {sum(p.count for p in forecast.points)} & {forecast.total_kg:.1f} & "
                 f"{forecast.ci_lo:.1f}--{forecast.ci_hi:.1f} \\\\",
                 r"\end{tabular}"]
        (out / "results_yield.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Отдельная оценка готового прогона
# ---------------------------------------------------------------------------
def evaluate_run(run_dir: str, gt_dir: str, cfg: Optional[Config] = None) -> dict:
    """Пересчитать метрики для уже сохранённого прогона (predictions.csv + gt)."""
    run = Path(run_dir)
    gt = load_ground_truth(gt_dir)
    cfg = cfg or Config()

    tracks_csv = run / "tracks.csv"
    per_frame_csv = run / "per_frame.csv"
    if not tracks_csv.exists():
        raise FileNotFoundError(f"в {run} нет tracks.csv — сначала выполните `applecounter run`")

    tracks = pd.read_csv(tracks_csv)
    counts_df = pd.read_csv(run / "counts.csv")
    pred_counts = {int(row["tree_id"]): int(row["count"]) for _, row in counts_df.iterrows()}
    if not pred_counts and "tree_id" in tracks.columns:
        pred_counts = {int(k): int(v) for k, v in tracks["tree_id"].value_counts().items()}

    count_m = counting_metrics(pred_counts, gt.counts,
                               tolerance=int(cfg.get("eval.count_tolerance", 3))) if gt.counts else None

    result = {
        "run": str(run),
        "gt": str(gt_dir),
        "count_metrics": count_m,
    }
    if per_frame_csv.exists() and gt.per_frame:
        logger.info("пер-кадровые метрики требуют предсказаний по кадрам — используйте `applecounter run --gt`")
    (run / "eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                                   encoding="utf-8")
    return result
