import numpy as np
import pytest

from applecounter.config import Config
from applecounter.trees.segment import build_tree_map, load_segments


def _cluster(cx, n=40, spread=8.0, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([rng.normal(cx, spread, n), rng.normal(0, 15, n)], axis=1)


def test_axis_three_clusters():
    pts = np.concatenate([_cluster(0, seed=1), _cluster(300, seed=2), _cluster(600, seed=3)])
    cfg = Config({"trees": {"method": "axis", "min_points": 10, "min_span": 15}})
    tm = build_tree_map(pts, cfg)
    assert tm.n_trees == 3
    ids = tm.assign_points(np.array([[10.0, 0], [310.0, 0], [610.0, 0]]))
    assert list(ids) == [1, 2, 3]
    # границы между кластерами (знак направления PCA произволен)
    assert len(tm.boundaries) == 2
    mids = sorted(abs(b - pts[:, 0].mean()) for b in tm.boundaries)
    assert mids[0] == pytest.approx(150, abs=60)
    assert mids[1] == pytest.approx(450, abs=60)


def test_axis_fixed_n_trees():
    pts = np.concatenate([_cluster(0), _cluster(300), _cluster(600), _cluster(900)])
    cfg = Config({"trees": {"method": "axis", "n_trees": 4, "min_points": 10}})
    tm = build_tree_map(pts, cfg)
    assert tm.n_trees == 4


def test_axis_too_few_points():
    cfg = Config({"trees": {"method": "axis"}})
    tm = build_tree_map(np.zeros((2, 2)), cfg)
    assert tm.n_trees == 1  # всё — одно «дерево» (нет границ)
    assert tm.meta.get("reason") in ("too_few_points", "zero_span", "no_intervals")


def test_assign_track_majority():
    cfg = Config({"trees": {"method": "axis", "n_trees": 2}})
    tm = build_tree_map(np.concatenate([_cluster(0), _cluster(400)]), cfg)
    points = np.array([[10.0, 0], [12.0, 1], [390.0, 0]])
    frames = [0, 1, 2]
    # большинство точек — в дереве 1
    assert tm.assign_track(points[:2], frames[:2]) == tm.assign_track(points, frames) in (1, 2)


def test_load_segments_named_video(tmp_path):
    p = tmp_path / "segments.csv"
    p.write_text("videoA,0,50,1\nvideoA,51,100,2\nvideoB,0,60,1\n", encoding="utf-8")
    segs = load_segments(str(p), video_name="videoA.mp4")
    assert segs == [(0, 50, 1), (51, 100, 2)]


def test_load_segments_plain_boundaries(tmp_path):
    p = tmp_path / "seg.csv"
    p.write_text("0,40,80\n", encoding="utf-8")
    segs = load_segments(str(p))
    assert segs == [(0, 40, 1), (40, 80, 2)]


def test_load_segments_wrong_video(tmp_path):
    p = tmp_path / "segments.csv"
    p.write_text("videoA,0,50,1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_segments(str(p), video_name="videoX.mp4")


def test_segments_method_build():
    p = "segments_tmp.csv"
    cfg = Config({"trees": {"method": "segments", "segments_file": None}})
    # с файлом
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as fh:
        fh.write("0,30,1\n31,60,2\n")
        name = fh.name
    try:
        cfg.set("trees.segments_file", name)
        tm = build_tree_map(np.zeros((10, 2)), cfg)
        assert tm.n_trees == 2
        assert tm.assign_frames(10) == 1
        assert tm.assign_frames(50) == 2
        assert tm.assign_frames(100) == 0
    finally:
        os.unlink(name)


def _row_detections(n_frames=1000, px_per_frame=2.0, box=75.0):
    """Детекции одного ряда: камера едет вдоль ряда, яблоко 75 px."""
    frames = np.arange(n_frames, dtype=float)
    cx = frames * px_per_frame
    half = box / 2.0
    return np.column_stack([cx - half, np.full(n_frames, 100.0 - half),
                            cx + half, np.full(n_frames, 100.0 + half), frames])


def test_subdivide_segments(tmp_path):
    import os
    seg = tmp_path / "seg.csv"
    seg.write_text("0,500,1\n500,1000,2\n", encoding="utf-8")
    det = _row_detections()
    cfg = Config({"trees": {"method": "segments", "segments_file": str(seg),
                            "spacing_m": 0.5, "apple_diameter_mm": 75.0}})
    pts = np.column_stack([(det[:, 0] + det[:, 2]) / 2, (det[:, 1] + det[:, 3]) / 2])
    tm = build_tree_map(pts, cfg, detections=det)

    # масштаб: 75 px на яблоко 75 мм -> 1000 px/м; шаг 0.5 м -> 500 px;
    # каждый stretch ~1000 px -> 2 дерева, всего 4
    assert tm.meta.get("subdivided") is True
    assert tm.meta["px_per_m"] == pytest.approx(1000, rel=0.01)
    assert tm.n_trees == 4
    assert len(tm.sub_edges) == 5

    # назначение по мировым координатам (ось ряда — X)
    assert tm.assign_track(np.array([[100.0, 100.0]]), [0]) == 1
    assert tm.assign_track(np.array([[1000.0, 100.0]]), [400]) == 2
    assert tm.assign_track(np.array([[1200.0, 100.0]]), [600]) == 3
    # до начала ряда — без привязки
    assert tm.assign_track(np.array([[-500.0, 100.0]]), [0]) == 0
    # кадровое назначение согласовано с мировым
    assert tm.assign_frames(100) == 1
    assert tm.assign_frames(900) == 4


def test_subdivide_disabled(tmp_path):
    seg = tmp_path / "seg.csv"
    seg.write_text("0,500,1\n500,1000,2\n", encoding="utf-8")
    det = _row_detections()
    cfg = Config({"trees": {"method": "segments", "segments_file": str(seg),
                            "subdivide": False}})
    pts = np.column_stack([(det[:, 0] + det[:, 2]) / 2, (det[:, 1] + det[:, 3]) / 2])
    tm = build_tree_map(pts, cfg, detections=det)
    assert tm.n_trees == 2
    assert len(tm.sub_edges) == 0
    # без sub_edges назначение идёт по кадрам (как раньше)
    assert tm.assign_track(np.array([[100.0, 100.0]]), [100]) == 1
    assert tm.assign_track(np.array([[100.0, 100.0]]), [900]) == 2


def test_subdivide_axis_sign_agnostic(tmp_path):
    """Направление PCA может быть любым — профиль должен расти по кадрам."""
    seg = tmp_path / "seg.csv"
    seg.write_text("0,1000,1\n", encoding="utf-8")
    # тот же ряд, но записан «справа налево» (убывание x с ростом кадра)
    det = _row_detections()
    x1 = 2000.0 - det[:, 2].copy()
    x2 = 2000.0 - det[:, 0].copy()
    det[:, 0], det[:, 2] = x1, x2
    cfg = Config({"trees": {"method": "segments", "segments_file": str(seg),
                            "spacing_m": 0.5, "apple_diameter_mm": 75.0}})
    pts = np.column_stack([(det[:, 0] + det[:, 2]) / 2, (det[:, 1] + det[:, 3]) / 2])
    tm = build_tree_map(pts, cfg, detections=det)
    assert tm.meta.get("subdivided") is True
    assert tm.n_trees >= 3  # ~2000 px / 500 px на дерево
