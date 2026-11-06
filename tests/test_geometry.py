import numpy as np
import pytest

from applecounter.utils import geometry as geo


def test_iou_matrix_basic():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]], dtype=np.float32)
    m = geo.iou_matrix(a, b)
    assert m.shape == (1, 3)
    assert m[0, 0] == pytest.approx(1.0)
    assert 0.0 < m[0, 1] < 1.0
    assert m[0, 2] == 0.0


def test_iou_empty():
    assert geo.iou_matrix(np.zeros((0, 4)), np.ones((3, 4))).shape == (0, 3)


def test_nms_keeps_best():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = geo.nms(boxes, scores, iou_thr=0.4)
    assert keep == [0, 2]


def test_shift_preserves_extra_columns():
    dets = np.array([[10, 20, 30, 40, 0.9, 1.0]], dtype=np.float32)
    out = geo.shift(dets, -5.0, 3.0)
    assert out.shape == (1, 6)
    np.testing.assert_allclose(out[0, :4], [5, 23, 25, 43])
    np.testing.assert_allclose(out[0, 4:], [0.9, 1.0], atol=1e-6)


def test_shift_empty():
    assert geo.shift(np.zeros((0, 6), dtype=np.float32), 1, 1).shape == (0, 6)
    assert geo.shift(np.zeros((0, 4), dtype=np.float32), 1, 1).shape == (0, 4)


def test_clip_boxes():
    boxes = np.array([[-5, -5, 105, 105]], dtype=np.float32)
    out = geo.clip_boxes(boxes, 100, 80)
    np.testing.assert_allclose(out, [[0, 0, 100, 80]])


def test_center_and_conversions():
    boxes = np.array([[0, 0, 10, 20]], dtype=np.float32)
    np.testing.assert_allclose(geo.center(boxes), [[5, 10]])
    cx = geo.xyxy_to_cxcywh(boxes)
    np.testing.assert_allclose(geo.cxcywh_to_xyxy(cx), boxes)
