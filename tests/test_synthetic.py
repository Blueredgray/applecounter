import json
from pathlib import Path

import pandas as pd

from applecounter.data.synthetic import SyntheticParams, generate_synthetic


def test_generate_synthetic(tmp_path):
    params = SyntheticParams(width=320, height=180, n_trees=2, apples_min=5, apples_max=8,
                             seed=3)  # frames=None -> автоматически на весь пролёт
    manifest = generate_synthetic(str(tmp_path), params)
    video = Path(manifest["video"])
    assert video.exists() and video.stat().st_size > 0

    gt = tmp_path / "gt"
    for name in ("gt_counts.csv", "objects.csv", "per_frame.csv", "segments.csv", "manifest.json"):
        assert (gt / name).exists(), name

    objects = pd.read_csv(gt / "objects.csv")
    counts = pd.read_csv(gt / "gt_counts.csv")
    assert len(objects) == sum(counts["count"])
    assert set(counts["tree_id"]) == {1, 2}

    per_frame = pd.read_csv(gt / "per_frame.csv")
    # каждое яблоко видно хотя бы в одном кадре
    seen = set(per_frame["apple_id"])
    assert set(objects["apple_id"]) <= seen
    # боксы внутри кадра
    assert (per_frame["x1"] >= -30).all() and (per_frame["x2"] <= params.width + 30).all()

    loaded = json.loads((gt / "manifest.json").read_text(encoding="utf-8"))
    assert loaded["n_apples"] == len(objects)


def test_counts_match_objects(tmp_path):
    params = SyntheticParams(width=320, height=180, n_trees=3, apples_min=4, apples_max=6,
                             frames=20, seed=11)
    generate_synthetic(str(tmp_path), params)
    objects = pd.read_csv(tmp_path / "gt" / "objects.csv")
    counts = pd.read_csv(tmp_path / "gt" / "gt_counts.csv")
    by_tree = objects.groupby("tree_id").size().to_dict()
    for _, row in counts.iterrows():
        assert int(row["count"]) == by_tree[int(row["tree_id"])]
