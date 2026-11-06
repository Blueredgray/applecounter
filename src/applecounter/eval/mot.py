"""Standard MOTChallenge metrics through the ``motmetrics`` package."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np


def motchallenge_metrics(
    pred_frames: Sequence[Mapping[int, np.ndarray]],
    gt_frames: Sequence[Mapping[int, np.ndarray]],
    iou_thr: float = 0.5,
) -> dict:
    try:
        import motmetrics as mm
    except ImportError as exc:
        raise ImportError("Install the 'motmetrics' package for standard MOT evaluation") from exc

    acc = mm.MOTAccumulator(auto_id=True)
    for pred_map, gt_map in zip(pred_frames, gt_frames):
        gt_ids = list(gt_map.keys())
        pred_ids = list(pred_map.keys())
        distances = np.full((len(gt_ids), len(pred_ids)), np.nan, dtype=float)
        if gt_ids and pred_ids:
            gt_boxes = np.asarray([gt_map[i] for i in gt_ids], dtype=np.float64)
            pred_boxes = np.asarray([pred_map[i] for i in pred_ids], dtype=np.float64)
            # motmetrics 1.4 calls NumPy's removed ``asfarray`` in its IoU helper.
            # Computing the same box IoU here keeps the metric backend usable with NumPy 2.
            lt = np.maximum(gt_boxes[:, None, :2], pred_boxes[None, :, :2])
            rb = np.minimum(gt_boxes[:, None, 2:4], pred_boxes[None, :, 2:4])
            wh = np.maximum(rb - lt, 0.0)
            intersection = wh[..., 0] * wh[..., 1]
            gt_area = np.prod(np.maximum(gt_boxes[:, 2:4] - gt_boxes[:, :2], 0.0), axis=1)
            pred_area = np.prod(np.maximum(pred_boxes[:, 2:4] - pred_boxes[:, :2], 0.0), axis=1)
            union = gt_area[:, None] + pred_area[None, :] - intersection
            iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
            distances = 1.0 - iou
            distances[distances > 1.0 - float(iou_thr)] = np.nan
        acc.update(gt_ids, pred_ids, distances)

    metrics = mm.metrics.motchallenge_metrics + ["num_matches", "idf1", "idtp", "idfp", "idfn"]
    metrics_host = mm.metrics.create()
    summary = metrics_host.compute(acc, metrics=metrics, name="sequence")
    row = summary.iloc[0]
    result = {
        "metric_backend": "motmetrics",
        "mota": float(row["mota"]),
        "idf1": float(row["idf1"]),
        "id_switches": int(row["num_switches"]),
        "tp": int(row["num_matches"]),
        "fp": int(row["num_false_positives"]),
        "fn": int(row["num_misses"]),
        "idtp": int(row["idtp"]),
        "idfp": int(row["idfp"]),
        "idfn": int(row["idfn"]),
        "n_gt": int(row["num_unique_objects"]),
        "iou_thr": float(iou_thr),
    }
    return result
