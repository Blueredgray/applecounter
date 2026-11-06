"""Общие фикстуры: src в sys.path, маленький синтетический датасет."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def tiny_synth(tmp_path_factory) -> dict:
    """Маленькое синтетическое видео (3 дерева) с ground truth."""
    from applecounter.data.synthetic import SyntheticParams, generate_synthetic

    out = tmp_path_factory.mktemp("synth")
    params = SyntheticParams(
        width=640, height=360, n_trees=3, apples_min=14, apples_max=18,
        seed=7,  # frames=None -> автоматически, чтобы пролёт покрыл все деревья
    )
    manifest = generate_synthetic(str(out), params)
    manifest["gt_dir"] = str(out / "gt")
    return manifest


@pytest.fixture(scope="session")
def tiny_run(tiny_synth, tmp_path_factory) -> dict:
    """Полный прогон пайплайна на маленьком синтетике (один раз на сессию)."""
    from applecounter.config import load_config
    from applecounter.pipeline import run_video

    out = tmp_path_factory.mktemp("run")
    cfg = load_config(overrides=["output.save_mosaic=false", "counting.merge_dist=14"])
    summary = run_video(cfg, tiny_synth["video"], str(out), gt_dir=tiny_synth["gt_dir"])
    summary["_out"] = str(out)
    return summary
