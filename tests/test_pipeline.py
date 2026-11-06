import json
from pathlib import Path

import pandas as pd

from applecounter.config import load_config


def test_pipeline_end_to_end(tiny_run, tiny_synth):
    summary = tiny_run
    out = Path(tiny_run["_out"])

    # -- артефакты -------------------------------------------------------
    assert (out / "summary.json").exists()
    assert (out / "report.md").exists()
    assert (out / "counts.csv").exists()
    assert (out / "tracks.csv").exists()
    assert (out / "per_frame.csv").exists()
    assert (out / "yield.json").exists()
    assert (out / "figures" / "counts_per_tree.png").exists()
    assert (out / "figures" / "pipeline.png").exists()

    # -- счёт ------------------------------------------------------------
    gt_counts = pd.read_csv(Path(tiny_synth["gt_dir"]) / "gt_counts.csv")
    gt_total = int(gt_counts["count"].sum())
    assert summary["n_trees"] == 3, summary["tree_meta"]
    total = summary["total_apples"]
    assert abs(total - gt_total) <= max(4, 0.15 * gt_total), f"{total} vs {gt_total}"
    assert summary["unassigned"] == 0

    # -- качество --------------------------------------------------------
    dm = summary["detection_metrics"]
    assert dm is not None and dm["ap50"] >= 0.6, dm
    cm = summary["count_metrics"]
    assert cm is not None
    assert abs(cm["total_error_pct"]) <= 15.0, cm
    # большинство деревьев в допуске
    assert cm["within_tolerance"] >= 2, cm

    # -- трекинг: без повторов -------------------------------------------
    tm = summary["tracking_metrics"]
    assert tm is not None and tm["metric_backend"] == "motmetrics", tm
    assert tm["mota"] >= 0.7 and tm["idf1"] >= 0.8, tm
    assert tm["id_switches"] <= tm["n_gt"] * 0.2, tm

    # -- прогноз урожая --------------------------------------------------
    y = summary["yield"]
    assert y is not None and y["total_kg"] > 0
    assert y["ci_lo"] <= y["total_kg"] <= y["ci_hi"]

    # -- стабилизация ----------------------------------------------------
    st = summary["stabilizer"]
    assert st["estimated"] > st["failed"]


def test_tracks_csv_unique_ids(tiny_run):
    tracks = pd.read_csv(Path(tiny_run["_out"]) / "tracks.csv")
    assert tracks["track_id"].is_unique
    assert (tracks["tree_id"] >= 0).all()
    assert set(tracks["tree_id"]) <= {0, 1, 2, 3}


def test_per_frame_monotonic(tiny_run):
    pf = pd.read_csv(Path(tiny_run["_out"]) / "per_frame.csv")
    diffs = pf["unique_counted"].diff().dropna()
    assert (diffs >= 0).all(), "накопленный счёт не должен убывать"


def test_report_contains_key_sections(tiny_run):
    text = Path(tiny_run["_out"], "report.md").read_text(encoding="utf-8")
    for marker in ("Отчёт", "Детекция", "Трекинг", "Счёт по деревьям", "Прогноз урожая"):
        assert marker in text, marker
    assert "AP@0.50" in text


def test_summary_json_serializable(tiny_run):
    raw = Path(tiny_run["_out"], "summary.json").read_text(encoding="utf-8")
    data = json.loads(raw)
    assert data["frames"] > 0
    assert data["detector"] == "hsv"


def test_evaluate_run_command(tiny_run, tiny_synth, tmp_path):
    from applecounter.pipeline import evaluate_run

    result = evaluate_run(tiny_run["_out"], tiny_synth["gt_dir"], load_config())
    assert result["count_metrics"] is not None
    assert (Path(tiny_run["_out"]) / "eval.json").exists()


def test_report_regeneration(tiny_run):
    """Команда report пересобирает отчёт из сохранённых артефактов."""
    from applecounter.cli import main

    code = main(["report", "--run", tiny_run["_out"]])
    assert code == 0
    assert (Path(tiny_run["_out"]) / "report.md").exists()


def test_yield_command(tiny_run, tmp_path):
    from applecounter.cli import main

    counts = Path(tiny_run["_out"]) / "counts.csv"
    code = main(["yield", "--counts", str(counts), "--out", str(tmp_path)])
    assert code == 0
    data = json.loads((tmp_path / "yield.json").read_text(encoding="utf-8"))
    assert data["total_kg"] > 0
