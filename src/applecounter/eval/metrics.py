"""Метрики: детекция (AP/PR), трекинг (IDSW/MOTA), точность счёта, регрессия."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..utils.geometry import iou_matrix


# ---------------------------------------------------------------------------
# Детекция
# ---------------------------------------------------------------------------
def match_frame(pred: np.ndarray, gt: np.ndarray, iou_thr: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Жадное сопоставление pred/gt по кадру в порядке убывания уверенности.

    pred, gt — (N, 6)/(M, 6) [x1, y1, x2, y2, conf, cls].
    Возвращает (bool TP для каждой pred, bool matched для каждой gt).
    """
    pred = np.asarray(pred, dtype=np.float32).reshape(-1, 6)
    gt = np.asarray(gt, dtype=np.float32).reshape(-1, 6)
    tp = np.zeros(len(pred), dtype=bool)
    gt_matched = np.zeros(len(gt), dtype=bool)
    if len(pred) == 0 or len(gt) == 0:
        return tp, gt_matched

    ious = iou_matrix(pred[:, :4], gt[:, :4])
    order = np.argsort(-pred[:, 4])
    for pi in order:
        gi = int(np.argmax(ious[pi]))
        if ious[pi, gi] >= iou_thr and not gt_matched[gi]:
            tp[pi] = True
            gt_matched[gi] = True
    return tp, gt_matched


def average_precision(scores: np.ndarray, is_tp: np.ndarray, n_gt: int) -> float:
    """AP по всем точкам интерполяции (в стиле VOC/COCO)."""
    scores = np.asarray(scores, dtype=float)
    is_tp = np.asarray(is_tp, dtype=bool)
    if n_gt == 0:
        return 0.0
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores)
    tp = is_tp[order].astype(float)
    fp = 1.0 - tp
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / n_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    # монотонное убывание точности справа налево
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    # интегрирование по recall
    prev_r = 0.0
    ap = 0.0
    for r, p in zip(recall, precision):
        ap += (r - prev_r) * p
        prev_r = r
    return float(ap)


def detection_metrics(
    preds: Sequence[np.ndarray],
    gts: Sequence[np.ndarray],
    iou_thr: float = 0.5,
) -> dict:
    """Метрики детекции по набору кадров (списки одинаковой длины)."""
    if len(preds) != len(gts):
        raise ValueError("preds и gts должны совпадать по длине")
    all_scores: List[float] = []
    all_tp: List[bool] = []
    n_gt = 0
    n_pred = 0
    n_tp = 0
    for pred, gt in zip(preds, gts):
        pred = np.asarray(pred, dtype=np.float32).reshape(-1, 6)
        gt = np.asarray(gt, dtype=np.float32).reshape(-1, 6)
        tp, gt_matched = match_frame(pred, gt, iou_thr)
        all_scores.extend(pred[:, 4].tolist() if len(pred) else [])
        all_tp.extend(tp.tolist())
        n_gt += len(gt)
        n_pred += len(pred)
        n_tp += int(tp.sum())

    ap = average_precision(np.asarray(all_scores), np.asarray(all_tp), n_gt)
    precision = n_tp / n_pred if n_pred else 0.0
    recall = n_tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "ap50": ap,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_gt": int(n_gt),
        "n_pred": int(n_pred),
        "tp": n_tp,
        "fp": int(n_pred - n_tp),
        "fn": int(n_gt - n_tp),
        "iou_thr": iou_thr,
    }


# ---------------------------------------------------------------------------
# Трекинг
# ---------------------------------------------------------------------------
def tracking_metrics(
    pred_frames: Sequence[Mapping[int, np.ndarray]],
    gt_frames: Sequence[Mapping[int, np.ndarray]],
    iou_thr: float = 0.5,
) -> dict:
    """Метрики трекинга: FP/FN/IDSW и MOTA.

    pred_frames/gt_frames: список по кадрам {id: box_xyxy}.
    """
    if len(pred_frames) != len(gt_frames):
        raise ValueError("число кадров в pred и gt должно совпадать")
    fp = fn = tp = idsw = 0
    gt_to_pred: Dict[int, int] = {}
    n_gt_total = 0
    n_pred_total = 0

    for pred_map, gt_map in zip(pred_frames, gt_frames):
        n_gt_total += len(gt_map)
        n_pred_total += len(pred_map)
        if not pred_map and not gt_map:
            continue
        pred_ids = list(pred_map.keys())
        gt_ids = list(gt_map.keys())
        if pred_ids and gt_ids:
            pred_boxes = np.asarray([pred_map[i] for i in pred_ids], dtype=np.float32)
            gt_boxes = np.asarray([gt_map[i] for i in gt_ids], dtype=np.float32)
            ious = iou_matrix(pred_boxes, gt_boxes)
            used_gt = set()
            for pi in np.argsort(-ious.max(axis=1) if ious.size else []):
                if not ious.size:
                    break
                gi = int(np.argmax(ious[pi]))
                if ious[pi, gi] < iou_thr or gi in used_gt:
                    continue
                used_gt.add(gi)
                p_id, g_id = pred_ids[int(pi)], gt_ids[gi]
                tp += 1
                prev = gt_to_pred.get(g_id)
                if prev is not None and prev != p_id:
                    # подмена id: объект был привязан к другому треку
                    idsw += 1
                gt_to_pred[g_id] = p_id
            fn += len(gt_ids) - len(used_gt)
            fp += len(pred_ids) - len(used_gt)
        else:
            fn += len(gt_ids)
            fp += len(pred_ids)

    denom = n_gt_total
    mota = 1.0 - (fn + fp + idsw) / denom if denom else 0.0
    return {
        "mota": float(mota),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "id_switches": idsw,
        "n_gt": n_gt_total,
        "n_pred": n_pred_total,
        "iou_thr": iou_thr,
    }


# ---------------------------------------------------------------------------
# Точность счёта
# ---------------------------------------------------------------------------
def counting_metrics(
    pred_counts: Mapping[int, int],
    gt_counts: Mapping[int, int],
    tolerance: int = 3,
) -> dict:
    """Метрики точности подсчёта по деревьям (+итог)."""
    keys = sorted(set(pred_counts) | set(gt_counts))
    rows = []
    abs_errors = []
    for key in keys:
        p = int(pred_counts.get(key, 0))
        g = int(gt_counts.get(key, 0))
        err = p - g
        abs_errors.append(err)
        rows.append({"tree_id": key, "pred": p, "gt": g, "error": err})

    pred_total = int(sum(pred_counts.values()))
    gt_total = int(sum(gt_counts.values()))
    arr = np.asarray(abs_errors, dtype=float)
    mae = float(np.abs(arr).mean()) if len(arr) else 0.0
    rmse = float(np.sqrt((arr ** 2).mean())) if len(arr) else 0.0
    bias = float(arr.mean()) if len(arr) else 0.0

    gt_vals = np.array([int(gt_counts.get(k, 0)) for k in keys], dtype=float)
    pred_vals = np.array([int(pred_counts.get(k, 0)) for k in keys], dtype=float)
    nonzero = gt_vals > 0
    mape = float(np.abs((pred_vals[nonzero] - gt_vals[nonzero]) / gt_vals[nonzero]).mean() * 100) if nonzero.any() else 0.0

    exact = sum(1 for r in rows if r["error"] == 0)
    within_tol = sum(1 for r in rows if abs(r["error"]) <= tolerance)
    total_err_pct = 100.0 * (pred_total - gt_total) / gt_total if gt_total else 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "mape": mape,
        "r2": r2_score(gt_vals, pred_vals),
        "pred_total": pred_total,
        "gt_total": gt_total,
        "total_error_pct": total_err_pct,
        "exact_trees": exact,
        "within_tolerance": within_tol,
        "n_trees": len(rows),
        "tolerance": tolerance,
        "per_tree": rows,
    }


def r2_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return 0.0
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot <= 0:
        return 1.0 if np.allclose(y_true, y_pred) else 0.0
    ss_res = float(((y_true - y_pred) ** 2).sum())
    return float(1.0 - ss_res / ss_tot)
