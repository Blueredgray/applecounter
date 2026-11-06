"""Экспорт прогона в формат VideoFruitCounting (all_predictions.csv)."""

import pandas as pd
import pytest

from applecounter.data.vfc import VFC_COLUMNS, export_vfc


def test_vfc_export_columns_and_geometry(tiny_run, tmp_path):
    out = export_vfc(tiny_run["_out"], out=str(tmp_path / "all_predictions.csv"))
    df = pd.read_csv(out)
    assert list(df.columns) == VFC_COLUMNS
    assert len(df) > 0
    assert (df["w"] > 0).all() and (df["h"] > 0).all()
    assert df["frame"].ge(0).all()
    assert df["track_id"].gt(0).all()
    assert df["score"].between(0, 1).all()


def test_vfc_export_default_layout(tiny_run):
    """Без --out файл ложится в <run>/vfc/<видео>/all_predictions.csv (формат assign_apples)."""
    out = export_vfc(tiny_run["_out"])
    parts = out.parts
    assert parts[-1] == "all_predictions.csv"
    assert parts[-3] == "vfc"
    assert out.exists()


def test_vfc_export_stretch_from_segments(tiny_run, tiny_synth, tmp_path):
    """SFRAME/segments: колонка stretch по кадровым границам дерева."""
    seg = str(tiny_synth["gt_dir"]) + "/segments.csv"
    out = export_vfc(tiny_run["_out"], out=str(tmp_path / "p.csv"), segments_file=seg)
    df = pd.read_csv(out)
    # 0 — кадры вне сегментов (у VideoFruitCounting такие тоже отбрасываются)
    assert set(df["stretch"].unique()) <= {0, 1, 2, 3}
    assert {1, 2, 3} & set(df["stretch"].unique())


def test_vfc_export_stretch_from_tracks(tmp_path, tiny_run):
    """Без segments колонка stretch берётся из tree_id в tracks.csv."""
    out = export_vfc(tiny_run["_out"], out=str(tmp_path / "p.csv"))
    df = pd.read_csv(out)
    assert set(df["stretch"].unique()) <= {0, 1, 2, 3}


def test_vfc_export_cli(tiny_run, tmp_path):
    from applecounter.cli import main

    target = tmp_path / "all_predictions.csv"
    code = main(["vfc-export", "--run", tiny_run["_out"], "--out", str(target)])
    assert code == 0
    assert target.exists()


def test_vfc_export_missing_run(tmp_path):
    with pytest.raises(FileNotFoundError):
        export_vfc(str(tmp_path))
