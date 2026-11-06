import json
from pathlib import Path

import pytest


def test_parser_all_commands():
    from applecounter.cli import build_parser

    parser = build_parser()
    # команды с обязательными аргументами
    for cmd in ("prepare", "train", "run", "evaluate", "yield", "report", "bench"):
        with pytest.raises(SystemExit):
            parser.parse_args([cmd])
    # команды без обязательных аргументов парсятся
    for cmd in ("synth", "download", "demo", "journals"):
        parser.parse_args([cmd])
    args = parser.parse_args(["run", "--video", "x.mp4"])
    assert args.command == "run" and args.video == "x.mp4"


def test_synth_command(tmp_path):
    from applecounter.cli import main

    code = main(["synth", "--out", str(tmp_path), "--trees", "2",
                 "--apples-min", "5", "--apples-max", "7", "--frames", "25",
                 "--width", "320", "--height", "180", "--seed", "5"])
    assert code == 0
    assert (tmp_path / "gt" / "gt_counts.csv").exists()
    assert list(tmp_path.glob("*.mp4"))


def test_journals_command(capsys):
    from applecounter.cli import main

    assert main(["journals"]) == 0
    out = capsys.readouterr().out
    assert "К1" in out and "К2" in out
    assert main(["journals", "--category", "К3"]) == 0
    out3 = capsys.readouterr().out
    assert "К3" in out3 and "К1" not in out3


def test_download_list(capsys):
    from applecounter.cli import main

    code = main(["download", "--list"])
    out = capsys.readouterr().out
    assert code == 0
    assert "acfr" in out and "minneapple" in out and "videofruit" in out
    assert "acfr-multifruit-2016.zip" in out


def test_download_unknown_dataset():
    from applecounter.cli import main

    assert main(["download", "--dataset", "нет"]) == 1


def test_run_missing_video(tmp_path):
    from applecounter.cli import main

    assert main(["run", "--video", str(tmp_path / "нет.mp4")]) == 1


def test_config_overrides_via_set(tmp_path):
    from applecounter.cli import build_parser
    from applecounter.config import load_config

    parser = build_parser()
    args = parser.parse_args(["run", "--video", "x.mp4", "--set", "tracker.n_init=4"])
    cfg = load_config(args.config, args.set)
    assert cfg.get("tracker.n_init") == 4


def test_bench_command_hsv(tmp_path):
    import cv2
    import numpy as np
    import yaml

    from applecounter.cli import main

    data_dir = tmp_path / "ds"
    (data_dir / "images" / "val").mkdir(parents=True)
    (data_dir / "labels" / "val").mkdir(parents=True)
    img = np.full((100, 100, 3), (40, 160, 40), dtype=np.uint8)  # зелёный фон
    cv2.circle(img, (50, 50), 20, (40, 40, 220), -1)  # красное яблоко
    cv2.imwrite(str(data_dir / "images" / "val" / "x.png"), img)
    (data_dir / "labels" / "val" / "x.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    (data_dir / "data.yaml").write_text(yaml.safe_dump({
        "path": str(data_dir), "train": "images/train", "val": "images/val",
        "nc": 1, "names": {0: "apple"},
    }), encoding="utf-8")

    out = tmp_path / "bench.json"
    code = main(["bench", "--data", str(data_dir), "--detector", "hsv", "--out", str(out)])
    assert code == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["n_images"] == 1
    m = result["hsv"]
    assert 0.0 <= m["ap50"] <= 1.0
    assert m["n_gt"] == 1 and m["count_mae_per_image"] >= 0.0
