"""Reproducible video/image-sequence pipeline: detection, tracking, counting and evaluation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import yaml

from .config import Config
from .counting.counter import AppleCounter, CountResult
from .data.mot import mot_identity_stats, read_mot_gt
from .detect.base import build_detector
from .detect.cache import DetectionCache
from .eval.metrics import counting_metrics, detection_metrics, tracking_metrics
from .eval.mot import motchallenge_metrics
from .track import build_tracker
from .track.tracker import TrackState
from .trees.segment import TreeMap, build_tree_map
from .trees.stabilize import Stabilizer
from .utils import geometry as geo
from .utils.logging import get_logger
from .utils.provenance import detection_cache_metadata, sha256_file, write_run_manifest
from .utils.video import VideoWriter, open_frame_source
from .viz import draw as viz_draw
from .viz import report as viz_report
from .yieldpred.model import YieldForecast, forecast_counts

logger = get_logger()
PIPELINE_VERSION = 2


@dataclass
class GroundTruth:
    counts: Dict[int, int] = field(default_factory=dict)
    per_frame: Dict[int, np.ndarray] = field(default_factory=dict)
    tracks: Dict[int, Dict[int, np.ndarray]] = field(default_factory=dict)
    objects: Optional[pd.DataFrame] = None
    source: str = ""
    identity_stats: dict = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return bool(self.counts)

    @property
    def tracking_valid(self) -> bool:
        return bool(self.tracks) and bool(self.identity_stats.get("tracking_gt_valid", False))


def load_ground_truth(gt_dir: str) -> GroundTruth:
    """Load internal CSV ground truth or a MOT Challenge ``gt/gt.txt`` dataset."""
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
    elif (root / "gt" / "gt.txt").exists():
        gt.per_frame, gt.tracks = read_mot_gt(root, zero_based=True)
    elif root.name == "gt.txt" and root.exists():
        gt.per_frame, gt.tracks = read_mot_gt(root, zero_based=True)

    if gt.tracks:
        gt.identity_stats = mot_identity_stats(gt.tracks)
        if not gt.identity_stats["tracking_gt_valid"]:
            logger.warning(
                "GT %s has no persistent identities; tracking metrics are disabled",
                root,
            )

    objects_path = root / "objects.csv"
    if objects_path.exists():
        gt.objects = pd.read_csv(objects_path)
    return gt


def _tree_map_for_run(cfg: Config, points: np.ndarray, detections: np.ndarray,
                      video_name: str) -> Optional[TreeMap]:
    if str(cfg.get("trees.method", "auto")).lower() == "none":
        return None
    return build_tree_map(points, cfg, video_name=video_name, detections=detections)


def run_video(cfg: Config, video_path: str, out_dir: str, gt_dir: Optional[str] = None) -> dict:
    """Run the complete pipeline and atomically save a self-describing result."""
    t_start = time.time()
    stride = int(cfg.get("video.frame_stride", 1))
    resize = cfg.get("video.resize")
    rotate90 = int(cfg.get("video.rotate90", 0))
    max_frames = cfg.get("video.max_frames")

    reader = open_frame_source(
        video_path, stride=stride, resize=tuple(resize) if resize else None,
        rotate90=rotate90, max_frames=max_frames,
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figures_dir = out / "figures"

    cache_path = cfg.get("detector.cache_file")
    cache_metadata = detection_cache_metadata(cfg, video_path) if cache_path else None
    detection_cache: Optional[DetectionCache] = None
    cache_builder: Optional[DetectionCache] = None
    cache_status = "disabled"
    if cache_path:
        if Path(cache_path).exists():
            detection_cache = DetectionCache.load(cache_path, expected_metadata=cache_metadata)
            cache_status = "loaded"
        else:
            cache_builder = DetectionCache(cache_metadata)
            cache_status = "created"

    detector = None if detection_cache is not None else build_detector(cfg)
    tracker = build_tracker(cfg)
    coordinate_system = getattr(tracker, "coordinate_system", "world")
    use_world = coordinate_system == "world"
    stabilizer = Stabilizer(cfg) if use_world else Stabilizer(Config({"stabilizer": {"enabled": False}}))
    counter = AppleCounter(min_obs=int(cfg.get("counting.min_obs", 2)))

    save_video = bool(cfg.get("output.save_video", True))
    annotate_ids = bool(cfg.get("output.annotate_ids", True))
    writer: Optional[VideoWriter] = None

    offsets: List[np.ndarray] = []
    world_points: List[np.ndarray] = []
    preds_by_frame: Dict[int, np.ndarray] = {}
    pred_tracks_by_frame: Dict[int, Dict[int, np.ndarray]] = {}
    frame_rows: List[dict] = []
    tracks_frame_rows: List[dict] = []
    counted_history: List[int] = []
    detection_seconds = 0.0
    tracking_seconds = 0.0

    gt: Optional[GroundTruth] = GroundTruth()
    if gt_dir:
        gt = load_ground_truth(gt_dir)
        if not gt.available:
            logger.warning("в %s нет gt_counts.csv — оценка точности счёта недоступна", gt_dir)

    n_processed = 0
    try:
        for src_idx, frame in reader.frames():
            cum = stabilizer.step(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) if use_world else np.zeros(2)
            t0 = time.time()
            if detection_cache is not None:
                if src_idx not in detection_cache:
                    raise ValueError(f"detection cache does not contain source frame {src_idx}")
                dets_screen = detection_cache.get(src_idx)
            else:
                dets_screen = detector.detect(frame)
                if cache_builder is not None:
                    cache_builder.add(src_idx, dets_screen)
            detection_seconds += time.time() - t0

            dets_tracker = geo.shift(dets_screen, -float(cum[0]), -float(cum[1])) if use_world else dets_screen
            if use_world and len(dets_tracker):
                world_points.append(np.column_stack([
                    dets_tracker[:, :4].astype(np.float64),
                    np.full(len(dets_tracker), float(n_processed)),
                ]))

            tracker_offset = (float(cum[0]), float(cum[1])) if use_world else (0.0, 0.0)
            t0 = time.time()
            views = tracker.update(n_processed, dets_tracker, image=frame, offset=tracker_offset)
            tracking_seconds += time.time() - t0
            counter.update(n_processed, views)

            preds_by_frame[src_idx] = dets_screen
            screen_tracks: Dict[int, np.ndarray] = {}
            for view in views:
                if view.state != TrackState.CONFIRMED or not view.updated:
                    continue
                box_screen = geo.shift(
                    np.asarray(view.box, dtype=np.float32).reshape(1, 4),
                    float(tracker_offset[0]), float(tracker_offset[1]),
                )[0]
                screen_tracks[view.track_id] = box_screen
                tracks_frame_rows.append({
                    "frame": src_idx, "track_id": view.track_id,
                    "x1": float(box_screen[0]), "y1": float(box_screen[1]),
                    "x2": float(box_screen[2]), "y2": float(box_screen[3]),
                    "conf": float(view.conf),
                })
            pred_tracks_by_frame[src_idx] = screen_tracks
            counted_history.append(len(counter._counted_ids))
            frame_rows.append({
                "frame": src_idx,
                "unique_counted": counted_history[-1],
                "active_tracks": int(sum(1 for v in views if v.state != TrackState.TENTATIVE)),
                "detections": int(len(dets_screen)),
                "cam_dx": float(cum[0]), "cam_dy": float(cum[1]),
            })
            offsets.append(np.asarray(cum, dtype=np.float64))

            if save_video:
                if writer is None:
                    writer = VideoWriter(str(out / "annotated.mp4"), reader.fps, reader.frame_size)
                hud = f"frame {src_idx} | tracks {counted_history[-1]}"
                writer.write(viz_draw.draw_tracks(
                    frame, views, offset=tracker_offset, annotate_ids=annotate_ids, hud_text=hud,
                ))

            n_processed += 1
            if n_processed % 100 == 0:
                logger.info("обработано кадров: %d | подтверждённых ID: %d", n_processed, len(counter._counted_ids))
    finally:
        if writer is not None:
            writer.close()
        reader.close()
        if detector is not None:
            detector.close()
        close_tracker = getattr(tracker, "close", None)
        if close_tracker is not None:
            close_tracker()

    if n_processed == 0:
        raise ValueError("no frames were processed")
    if cache_builder is not None:
        cache_builder.save(cache_path)
        cache_sha256 = sha256_file(cache_path)
    elif detection_cache is not None:
        cache_sha256 = sha256_file(cache_path)
    else:
        cache_sha256 = None

    points_arr = np.concatenate(world_points, axis=0) if world_points else np.zeros((0, 5))
    tree_map = _tree_map_for_run(
        cfg, points_arr[:, :2] if len(points_arr) else points_arr,
        points_arr if len(points_arr) else None, Path(video_path).name,
    )
    counts: CountResult = counter.finalize(
        tree_map, tracker.tracks, merge_dist=float(cfg.get("counting.merge_dist", 0.0)),
    )
    count_unit = "stretch" if tree_map is not None and tree_map.method in ("segments", "frames") else "tree"
    if tree_map is None:
        count_unit = "sequence"
    if count_unit == "stretch" and str(cfg.get("trees.assignment", "median_frame")) == "vfc_reference":
        from .trees.vfc_assignment import assign_vfc_stretches

        marker_frames = [tree_map.frame_segments[0][0]] + [seg[1] for seg in tree_map.frame_segments]
        counted_ids = {rec.track_id for rec in counts.records}
        observations: Dict[int, List[tuple[int, np.ndarray]]] = {}
        for row in tracks_frame_rows:
            tid = int(row["track_id"])
            if tid in counted_ids:
                observations.setdefault(tid, []).append((
                    int(row["frame"]),
                    np.asarray([row["x1"], row["y1"], row["x2"], row["y2"]], dtype=float),
                ))
        assignments, assignment_meta = assign_vfc_stretches(
            observations, marker_frames, reader.frame_size[0], Path(video_path).name,
        )
        per_stretch: Dict[int, int] = {}
        unassigned = 0
        for rec in counts.records:
            rec.tree_id = int(assignments.get(rec.track_id, 0))
            if rec.tree_id > 0:
                per_stretch[rec.tree_id] = per_stretch.get(rec.tree_id, 0) + 1
            else:
                unassigned += 1
        counts.per_tree = dict(sorted(per_stretch.items()))
        counts.unassigned = unassigned
        tree_map.meta.update(assignment_meta)
    logger.info("единица счёта: %s; найдено треков после фильтра: %d", count_unit, counts.total)

    yield_input = {k: v for k, v in counts.per_tree.items() if k > 0}
    forecast: Optional[YieldForecast] = None
    if yield_input and bool(cfg.get("yield.enabled", True)):
        try:
            forecast = forecast_counts(yield_input, cfg)
        except Exception as exc:
            logger.warning("прогноз урожая не рассчитан: %s", exc)

    det_metrics = track_metrics = count_metrics = None
    if gt is not None and (gt.per_frame or gt.counts):
        common = sorted(set(preds_by_frame) & set(gt.per_frame))
        if common:
            det_metrics = detection_metrics(
                [preds_by_frame[f] for f in common], [gt.per_frame[f] for f in common],
                iou_thr=float(cfg.get("eval.iou_match", 0.5)),
            )
            if gt.tracking_valid:
                pred_eval = [pred_tracks_by_frame.get(f, {}) for f in common]
                gt_eval = [gt.tracks[f] for f in common]
                try:
                    track_metrics = motchallenge_metrics(
                        pred_eval, gt_eval, iou_thr=float(cfg.get("eval.iou_match", 0.5)),
                    )
                except ImportError:
                    logger.warning("motmetrics is unavailable; using internal MOTA fallback")
                    track_metrics = tracking_metrics(pred_eval, gt_eval)
                    track_metrics["metric_backend"] = "internal"
                track_metrics["n_pred"] = int(sum(len(x) for x in pred_eval))
        if gt.counts:
            count_metrics = counting_metrics(
                {k: v for k, v in counts.per_tree.items() if k > 0}, gt.counts,
                tolerance=int(cfg.get("eval.count_tolerance", 3)),
            )

    tracker_summary = {
        "name": str(cfg.get("tracker.name", "ours")),
        "implementation": getattr(tracker, "implementation", tracker.__class__.__name__),
        "coordinate_system": coordinate_system,
        "unique_counted": counts.total,
        "created_tracks": tracker.n_created,
        "n_double_count_events": counter.n_double_count_events,
        "double_count_metric_scope": "repeated count_event for the same internal ID only; not physical double counting",
    }
    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "video": str(video_path),
        "frames": n_processed,
        "detector": str(cfg.get("detector.name", "hsv")),
        "detector_params": {
            "name": str(cfg.get("detector.name", "hsv")),
            "weights": cfg.get("detector.yolo.weights") if str(cfg.get("detector.name", "hsv")) == "yolo" else None,
            "imgsz": cfg.get("detector.yolo.imgsz") if str(cfg.get("detector.name", "hsv")) == "yolo" else None,
            "conf": cfg.get("detector.yolo.conf") if str(cfg.get("detector.name", "hsv")) == "yolo" else None,
        },
        "detection_cache": {
            "path": str(cache_path) if cache_path else None,
            "status": cache_status,
            "sha256": cache_sha256,
        },
        "stabilizer": {
            "enabled": bool(use_world and cfg.get("stabilizer.enabled", True)),
            "estimated": stabilizer.n_estimated,
            "failed": stabilizer.n_failed,
            "path_length_px": round(stabilizer.path_length, 1),
        },
        "tracker": tracker_summary,
        "count_unit": count_unit,
        "n_trees": tree_map.n_trees if tree_map is not None else 0,
        "tree_method": tree_map.method if tree_map is not None else "none",
        "tree_meta": ({**tree_map.meta,
                       "image_width": float(reader.frame_size[0]),
                       "boundaries": [float(b) for b in np.atleast_1d(tree_map.boundaries)],
                       "origin": [float(v) for v in tree_map.origin],
                       "direction": [float(v) for v in tree_map.direction]}
                      if tree_map is not None else {}),
        "counts": {str(k): v for k, v in counts.per_tree.items() if k > 0},
        "unassigned": counts.unassigned,
        "total_apples": counts.total,
        "yield": None if forecast is None else {
            "method": forecast.method,
            "estimate_type": forecast.estimate_type,
            "total_kg": round(forecast.total_kg, 2),
            "sigma_kg": round(forecast.total_sigma, 2),
            "ci_lo": round(forecast.ci_lo, 2),
            "ci_hi": round(forecast.ci_hi, 2),
            "params": forecast.params,
        },
        "detection_metrics": det_metrics,
        "tracking_metrics": track_metrics,
        "count_metrics": count_metrics,
        "gt_identity_stats": gt.identity_stats if gt is not None else {},
        "timing": {
            "elapsed_sec": round(time.time() - t_start, 2),
            "detection_sec": round(detection_seconds, 2),
            "tracking_sec": round(tracking_seconds, 2),
        },
        "elapsed_sec": round(time.time() - t_start, 2),
    }

    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg.data, allow_unicode=True, sort_keys=False), encoding="utf-8",
    )
    pd.DataFrame(counts.as_rows(), columns=[
        "track_id", "tree_id", "first_frame", "last_frame", "n_obs", "x", "y", "mean_conf",
    ]).to_csv(out / "tracks.csv", index=False)
    pd.DataFrame(tracks_frame_rows, columns=[
        "frame", "track_id", "x1", "y1", "x2", "y2", "conf",
    ]).to_csv(out / "tracks_frame.csv", index=False)
    pd.DataFrame(frame_rows).to_csv(out / "per_frame.csv", index=False)
    pd.DataFrame([
        {"tree_id": k, "count": v, "unit": count_unit}
        for k, v in counts.per_tree.items() if k > 0
    ], columns=["tree_id", "count", "unit"]).to_csv(out / "counts.csv", index=False)
    if forecast is not None:
        pd.DataFrame(forecast.as_rows()).to_csv(out / "yield.csv", index=False)
        (out / "yield.json").write_text(json.dumps({
            "method": forecast.method,
            "estimate_type": forecast.estimate_type,
            "total_kg": forecast.total_kg,
            "total_sigma": forecast.total_sigma,
            "ci": [forecast.ci_lo, forecast.ci_hi],
            "params": forecast.params,
            "per_tree": forecast.as_rows(),
        }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    figures: Dict[str, str] = {}
    if bool(cfg.get("output.save_figures", True)):
        figures_dir.mkdir(exist_ok=True)
        figures[f"Подсчёт по {viz_report.unit_label(count_unit, plural=True)}"] = \
            viz_report.fig_counts_per_tree(
                counts, gt.counts if gt and gt.counts else None,
                figures_dir / "counts_per_tree.png", count_unit=count_unit,
            ).name
        figures["Накопленный счёт"] = viz_report.fig_cumulative(
            counts.per_frame_ids, figures_dir / "cumulative_count.png",
        ).name
        if det_metrics is not None:
            figures["Кривая PR"] = viz_report.fig_pr(
                [preds_by_frame[f] for f in sorted(set(preds_by_frame) & set(gt.per_frame))],
                [gt.per_frame[f] for f in sorted(set(preds_by_frame) & set(gt.per_frame))],
                figures_dir / "pr_curve.png", iou_thr=float(cfg.get("eval.iou_match", 0.5)),
            ).name
        if count_metrics is not None:
            figures[f"Ошибка по {viz_report.unit_label(count_unit, plural=True)}"] = \
                viz_report.fig_error_per_tree(
                    count_metrics["per_tree"], figures_dir / "error_per_tree.png",
                    count_unit=count_unit,
                ).name
        if forecast is not None and forecast.points:
            figures["Прогноз урожая"] = viz_report.fig_yield(
                forecast, figures_dir / "yield_forecast.png", count_unit=count_unit,
            ).name
        figures["Структура метода"] = viz_report.fig_pipeline(figures_dir / "pipeline.png").name

    if bool(cfg.get("output.save_mosaic", True)) and offsets and use_world:
        try:
            mosaic = viz_report.make_mosaic_figure(
                video_path, np.stack(offsets, axis=0), cfg, counts.records,
                out / "mosaic.png", tree_map=tree_map,
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
    write_latex_assets(out, summary, count_metrics, det_metrics, forecast)
    summary["paths"] = {
        "report": str(report_path), "summary": str(out / "summary.json"),
        "annotated": str(out / "annotated.mp4") if (out / "annotated.mp4").exists() else None,
        "figures": {k: str(out / "figures" / v) for k, v in figures.items()},
    }
    write_run_manifest(out / "run_manifest.json", cfg, video_path, status="complete")
    return summary


def write_latex_assets(out: Path, summary: dict, count_metrics: Optional[dict],
                       det_metrics: Optional[dict], forecast: Optional[YieldForecast]) -> None:
    lines = ["% generated by applecounter", r"\begin{tabular}{lcc}", r"\hline",
             "Показатель & Значение \\", r"\hline"]
    if det_metrics:
        lines += [
            f"AP@{det_metrics['iou_thr']:.2f} & {det_metrics['ap50']:.3f} \\",
            f"Precision / Recall & {det_metrics['precision']:.3f} / {det_metrics['recall']:.3f} \\",
            f"F1 & {det_metrics['f1']:.3f} \\",
        ]
    tm = summary.get("tracking_metrics")
    if tm:
        lines.append(f"MOTA & {tm['mota']:.3f} \\")
        if "idf1" in tm:
            lines.append(f"IDF1 & {tm['idf1']:.3f} \\")
        lines.append(f"Подмены ID & {tm['id_switches']} \\")
    if count_metrics:
        lines += [
            f"Ошибка итогового счёта, \\% & {count_metrics['total_error_pct']:.1f} \\",
            f"MAE (на единицу) & {count_metrics['mae']:.2f} \\",
            f"RMSE & {count_metrics['rmse']:.2f} \\",
            f"MAPE, \\% & {count_metrics['mape']:.1f} \\",
        ]
    lines += [r"\hline", r"\end{tabular}"]
    (out / "results_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if forecast is not None:
        rows = ["% generated by applecounter", r"\begin{tabular}{lrrr}", r"\hline",
                "Единица & Яблок & Масса, кг & 95\\% ДИ, кг \\", r"\hline"]
        for point in forecast.points:
            rows.append(f"{point.tree_id} & {point.count} & {point.mass_kg:.1f} & {point.ci_lo:.1f}--{point.ci_hi:.1f} \\")
        rows += [r"\hline",
                 f"Итого & {sum(p.count for p in forecast.points)} & {forecast.total_kg:.1f} & "
                 f"{forecast.ci_lo:.1f}--{forecast.ci_hi:.1f} \\", r"\end{tabular}"]
        (out / "results_yield.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


def evaluate_run(run_dir: str, gt_dir: str, cfg: Optional[Config] = None,
                 out_path: Optional[str] = None) -> dict:
    """Evaluate saved raw tracks/counts without rewriting the original run by default."""
    run = Path(run_dir)
    gt = load_ground_truth(gt_dir)
    cfg = cfg or Config()
    tracks = pd.read_csv(run / "tracks.csv")
    counts_df = pd.read_csv(run / "counts.csv")
    pred_counts = {int(row["tree_id"]): int(row["count"]) for _, row in counts_df.iterrows()}
    if not pred_counts and "tree_id" in tracks.columns:
        pred_counts = {int(k): int(v) for k, v in tracks["tree_id"].value_counts().items()}
    count_m = counting_metrics(
        pred_counts, gt.counts, tolerance=int(cfg.get("eval.count_tolerance", 3)),
    ) if gt.counts else None
    result = {"run": str(run), "gt": str(gt_dir), "count_metrics": count_m,
              "tracking_metrics": None, "gt_identity_stats": gt.identity_stats}

    tracks_frame = run / "tracks_frame.csv"
    if tracks_frame.exists() and gt.tracking_valid:
        df = pd.read_csv(tracks_frame)
        pred_frames: Dict[int, Dict[int, np.ndarray]] = {}
        for row in df.itertuples(index=False):
            pred_frames.setdefault(int(row.frame), {})[int(row.track_id)] = np.asarray(
                [row.x1, row.y1, row.x2, row.y2], dtype=np.float32,
            )
        common = sorted(set(pred_frames) & set(gt.tracks))
        # Include GT-only frames so misses are counted.
        common = sorted(set(common) | set(gt.tracks))
        result["tracking_metrics"] = motchallenge_metrics(
            [pred_frames.get(f, {}) for f in common], [gt.tracks[f] for f in common],
            iou_thr=float(cfg.get("eval.iou_match", 0.5)),
        )
    target = Path(out_path) if out_path else run / "eval.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return result
