import numpy as np

from applecounter.eval.metrics import (
    average_precision,
    counting_metrics,
    detection_metrics,
    match_frame,
    r2_score,
    tracking_metrics,
)


def _det(x1, y1, conf=0.9, cls=0.0):
    return [x1, y1, x1 + 10, y1 + 10, conf, cls]


def test_match_frame_perfect():
    pred = np.array([_det(0, 0), _det(50, 0)], dtype=np.float32)
    gt = np.array([_det(0, 0, 1.0), _det(50, 0, 1.0)], dtype=np.float32)
    tp, matched = match_frame(pred, gt, iou_thr=0.5)
    assert tp.all() and matched.all()


def test_detection_metrics_perfect():
    frames_pred = [np.array([_det(0, 0), _det(30, 30)], dtype=np.float32)] * 3
    frames_gt = [np.array([_det(0, 0), _det(30, 30)], dtype=np.float32)] * 3
    m = detection_metrics(frames_pred, frames_gt, iou_thr=0.5)
    assert m["ap50"] == 1.0
    assert m["precision"] == 1.0 and m["recall"] == 1.0
    assert m["fp"] == 0 and m["fn"] == 0


def test_detection_metrics_false_positive():
    pred = [np.array([_det(0, 0), _det(90, 90, conf=0.4)], dtype=np.float32)]
    gt = [np.array([_det(0, 0)], dtype=np.float32)]
    m = detection_metrics(pred, gt, iou_thr=0.5)
    assert m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 0
    assert 0.9 < m["ap50"] <= 1.0
    assert m["precision"] == 0.5


def test_average_precision_empty():
    assert average_precision(np.array([]), np.array([]), n_gt=0) == 0.0


def test_tracking_metrics_id_switch():
    # GT-объект 1: первый кадр привязан к треку 10, второй — к треку 20
    pred = [
        {10: np.array([0, 0, 10, 10], dtype=np.float32)},
        {20: np.array([0, 0, 10, 10], dtype=np.float32)},
    ]
    gt = [
        {1: np.array([0, 0, 10, 10], dtype=np.float32)},
        {1: np.array([0, 0, 10, 10], dtype=np.float32)},
    ]
    m = tracking_metrics(pred, gt)
    assert m["id_switches"] == 1
    assert m["mota"] < 1.0


def test_tracking_metrics_perfect():
    frames = [{1: np.array([0, 0, 10, 10], dtype=np.float32),
               2: np.array([20, 0, 30, 10], dtype=np.float32)}] * 5
    m = tracking_metrics(frames, frames)
    assert m["mota"] == 1.0
    assert m["id_switches"] == 0 and m["fp"] == 0 and m["fn"] == 0


def test_counting_metrics():
    pred = {1: 20, 2: 30, 3: 10}
    gt = {1: 20, 2: 28, 3: 12}
    m = counting_metrics(pred, gt, tolerance=3)
    assert m["pred_total"] == 60 and m["gt_total"] == 60
    assert m["total_error_pct"] == 0.0
    assert m["mae"] == (0 + 2 + 2) / 3
    assert m["within_tolerance"] == 3 and m["n_trees"] == 3
    assert m["r2"] <= 1.0
    rows = {r["tree_id"]: r for r in m["per_tree"]}
    assert rows[2]["error"] == 2


def test_r2_score():
    assert r2_score([1, 2, 3], [1, 2, 3]) == 1.0
    assert r2_score([1, 2, 3], [3, 2, 1]) < 1.0
    assert r2_score([], []) == 0.0
