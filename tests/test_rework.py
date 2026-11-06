import json

import cv2
import numpy as np
import pytest

from applecounter.config import Config, load_config
from applecounter.detect.cache import DetectionCache
from applecounter.eval.mot import motchallenge_metrics
from applecounter.pipeline import load_ground_truth


def _box(x=0.0, y=0.0):
    return np.asarray([x, y, x + 10, y + 10], dtype=np.float32)


def test_detection_cache_roundtrip_and_mismatch(tmp_path):
    path = tmp_path / "detections.npz"
    cache = DetectionCache({"detector": "fixed", "source_sha256": "abc"})
    cache.add(2, np.asarray([[1, 2, 3, 4, 0.8, 0]], dtype=np.float32))
    cache.add(0, np.zeros((0, 6), dtype=np.float32))
    cache.save(path)

    loaded = DetectionCache.load(path, expected_metadata={
        "detector": "fixed", "source_sha256": "abc",
    })
    assert list(loaded) == [0, 2]
    assert loaded.get(2).shape == (1, 6)
    assert loaded.get(0).shape == (0, 6)
    assert loaded.metadata["n_frames"] == 2
    with pytest.raises(ValueError, match="configuration mismatch"):
        DetectionCache.load(path, expected_metadata={"detector": "other"})


def test_mot_gt_and_nonpersistent_minneapple_ids(tmp_path):
    mot = tmp_path / "mot"
    (mot / "gt").mkdir(parents=True)
    (mot / "gt" / "gt.txt").write_text(
        "1,7,10,20,30,40,1,1,1\n"
        "2,7,11,21,30,40,1,1,1\n"
        "2,8,50,50,20,20,0,1,1\n",
        encoding="utf-8",
    )
    gt = load_ground_truth(str(mot))
    assert gt.tracking_valid
    assert set(gt.tracks) == {0, 1}
    assert gt.identity_stats["n_ids"] == 1
    assert gt.identity_stats["n_observations"] == 2
    np.testing.assert_allclose(gt.tracks[0][7], [10, 20, 40, 60])

    ma = tmp_path / "minneapple"
    ma.mkdir()
    (ma / "per_frame.csv").write_text(
        "frame,apple_id,x1,y1,x2,y2\n"
        "0,1,0,0,10,10\n"
        "1,2,0,0,10,10\n"
        "2,3,0,0,10,10\n",
        encoding="utf-8",
    )
    gt_ma = load_ground_truth(str(ma))
    assert not gt_ma.tracking_valid
    assert gt_ma.identity_stats["n_repeated_ids"] == 0


def test_motmetrics_perfect_and_id_switch():
    gt = [{1: _box()}, {1: _box()}]
    perfect = motchallenge_metrics([{10: _box()}, {10: _box()}], gt)
    assert perfect["mota"] == pytest.approx(1.0)
    assert perfect["idf1"] == pytest.approx(1.0)
    assert perfect["id_switches"] == 0
    switched = motchallenge_metrics([{10: _box()}, {20: _box()}], gt)
    assert switched["id_switches"] == 1
    assert switched["idf1"] < 1.0


def test_pipeline_reuses_identical_detection_cache(tmp_path, monkeypatch):
    from applecounter.pipeline import run_video

    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(4):
        image = np.full((80, 100, 3), (40, 160, 40), dtype=np.uint8)
        cv2.circle(image, (30 + i, 40), 8, (40, 40, 220), -1)
        cv2.imwrite(str(frames / f"{i:06d}.png"), image)
    cache_path = tmp_path / "detections.npz"
    cfg = load_config(overrides=[
        "trees.method=none", "output.save_video=false", "output.save_figures=false",
        "output.save_mosaic=false", "yield.enabled=false", f"detector.cache_file={cache_path}",
    ])
    first = run_video(cfg, str(frames), str(tmp_path / "first"))
    assert first["detection_cache"]["status"] == "created"

    def forbidden_detector(_cfg):
        raise AssertionError("detector must not be constructed when cache is loaded")

    monkeypatch.setattr("applecounter.pipeline.build_detector", forbidden_detector)
    second = run_video(cfg, str(frames), str(tmp_path / "second"))
    assert second["detection_cache"]["status"] == "loaded"
    assert first["detection_cache"]["sha256"] == second["detection_cache"]["sha256"]
    assert first["frames"] == second["frames"] == 4


def test_external_botsort_bytetrack_and_cnn_deepsort():
    from applecounter.track import build_tracker

    image = np.full((120, 160, 3), 40, dtype=np.uint8)
    detections = np.asarray([[30, 30, 60, 60, 0.9, 0]], dtype=np.float32)
    for name in ("botsort", "bytetrack"):
        cfg = Config()
        cfg.set("tracker.name", name)
        cfg.set("tracker.n_init", 1)
        cfg.set("tracker.use_embeddings", name == "botsort")
        cfg.set("tracker.cmc_enabled", False)
        tracker = build_tracker(cfg)
        views = [tracker.update(i, detections, image=image) for i in range(3)]
        assert views[0] and views[0][0].count_event
        assert views[2][0].track_id == views[0][0].track_id
        assert views[2][0].hits == 3
        assert not views[2][0].count_event

    cfg = Config()
    cfg.set("tracker.name", "deepsort")
    cfg.set("tracker.n_init", 1)
    cfg.set("project.device", "cpu")
    tracker = build_tracker(cfg)
    try:
        views = [tracker.update(i, detections, image=image) for i in range(4)]
        assert views[1] and views[1][0].count_event
        assert views[3][0].track_id == views[1][0].track_id
        assert views[3][0].hits == 3
    finally:
        tracker.close()


def test_vfc_reference_marker_assignment_and_west_reversal():
    from applecounter.trees.vfc_assignment import assign_vfc_stretches

    observations = {
        10: [(10, [0, 0, 20, 20])],       # left of marker 1 -> stretch 0 -> unassigned
        11: [(10, [80, 0, 100, 20])],    # right of marker 1 -> stretch 1
        12: [(20, [0, 0, 20, 20])],      # left of marker 2 -> stretch 1
        13: [(20, [80, 0, 100, 20])],    # right of marker 2 -> stretch 2
        14: [(30, [80, 0, 100, 20])],    # after the last stretch -> unassigned
    }
    assigned, meta = assign_vfc_stretches(observations, [10, 20, 30], 100, "row_e")
    assert assigned == {10: 0, 11: 1, 12: 1, 13: 2, 14: 0}
    assert meta["assignment"] == "vfc_reference"
    west, _ = assign_vfc_stretches(observations, [10, 20, 30], 100, "a_b_c_d_w_e")
    assert west[11] == 2 and west[13] == 1


def test_retree_preserves_vfc_reference_and_disables_yield(tmp_path):
    import pandas as pd

    from applecounter.cli import main

    run = tmp_path / "run"
    run.mkdir()
    segments = tmp_path / "sframe.csv"
    segments.write_text("row_e,10,20,30\n", encoding="utf-8")
    (run / "tracks.csv").write_text(
        "track_id,first_frame,last_frame,n_obs,x,y,mean_conf,tree_id\n"
        "1,10,10,1,90,10,0.9,0\n",
        encoding="utf-8",
    )
    (run / "tracks_frame.csv").write_text(
        "frame,track_id,x1,y1,x2,y2,conf\n"
        "10,1,80,0,100,20,0.9\n",
        encoding="utf-8",
    )
    (run / "per_frame.csv").write_text(
        "frame,unique_counted,active_tracks,detections,cam_dx,cam_dy\n"
        "10,1,1,1,0,0\n",
        encoding="utf-8",
    )
    summary = {
        "video": "row_e.mp4", "frames": 1, "total_apples": 1,
        "tree_method": "segments", "count_unit": "stretch", "unassigned": 0,
        "tree_meta": {
            "file": str(segments), "assignment": "vfc_reference",
            "image_width": 100.0, "boundaries": [], "origin": [0, 0],
            "direction": [1, 0], "sub_edges": [],
        },
    }
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    code = main([
        "retree", "--run", str(run),
        "--set", "trees.assignment=median_frame",
        "--set", "yield.enabled=false",
    ])
    assert code == 0
    updated = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert updated["tree_meta"]["assignment"] == "vfc_reference"
    assert updated["count_unit"] == "stretch"
    assert updated["counts"] == {"1": 1}
    assert updated["yield"] is None
    counts = pd.read_csv(run / "counts.csv")
    assert counts.to_dict("records") == [{"tree_id": 1, "count": 1, "unit": "stretch"}]
    assert not (run / "yield.csv").exists()
    assert not (run / "yield.json").exists()
    report = (run / "report.md").read_text(encoding="utf-8")
    assert "Счёт по stretch" in report
    assert "Счёт по деревьям" not in report
    assert "IDF1" not in report or "## Трекинг" in report


def test_yield_cli_accepts_external_validation_csv(tmp_path):
    import pandas as pd

    from applecounter.cli import main

    train = tmp_path / "field_train.csv"
    train.write_text("tree_id,count,mass_kg\n1,10,2\n2,20,4\n3,30,6\n4,40,8\n",
                     encoding="utf-8")
    holdout = tmp_path / "field_holdout.csv"
    holdout.write_text("tree_id,count,mass_kg\n5,15,3\n", encoding="utf-8")
    counts = tmp_path / "counts.csv"
    pd.DataFrame([{"tree_id": 1, "count": 20, "unit": "tree"}]).to_csv(counts, index=False)
    out = tmp_path / "yield_out"

    code = main([
        "yield", "--counts", str(counts), "--out", str(out),
        "--regression-csv", str(train), "--validation-csv", str(holdout),
    ])
    assert code == 0
    result = json.loads((out / "yield.json").read_text(encoding="utf-8"))
    assert result["method"] == "regression"
    assert result["estimate_type"] == "field_calibrated_external_holdout"
    assert result["params"]["validation"]["method"] == "external_holdout"
