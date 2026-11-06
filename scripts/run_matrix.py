"""Полная матрица комбинаций: детекторы x трекеры x датасеты.

VFC (16 видео): hsv|yolo8n|yolo26s x botsort|deepsort.
  Комбинация yolo26s+botsort уже выполнена в outputs/vfc_<video> — используем её.
  Остальные идут в outputs/matrix/<combo>/<video>.
MinneApple (видео 100 кадров с GT): hsv x botsort|deepsort (остальные уже есть).

Пропуск: если summary.json в целевом каталоге уже есть.
Запуск: python scripts/run_matrix.py [--dry-run] [--only <подстрока>]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VFC_ROOT = ROOT / "data" / "raw" / "videofruit" / "Video Fruit Counting"
SFRAME = VFC_ROOT / "stretch_sframe.csv"
VIDEOS = VFC_ROOT / "Videos"
MN_VIDEO = ROOT / "data" / "processed" / "minneapple_video" / "20150919_174730.mp4"
MN_GT = ROOT / "data" / "processed" / "minneapple_video" / "gt"
OUT = ROOT / "outputs"
MATRIX = OUT / "matrix"
LOG = OUT / "matrix.log"

WEIGHT_8N = "weights/runs/minneapple50/weights/best.pt"
WEIGHT_26S = "weights/runs/minneapple26s1280/weights/best.pt"

COMBOS = {
    "hsv_botsort": {"detector": "hsv", "tracker": "botsort"},
    "hsv_deepsort": {"detector": "hsv", "tracker": "deepsort"},
    "yolo8n_botsort": {"detector": "yolo8n", "tracker": "botsort"},
    "yolo8n_deepsort": {"detector": "yolo8n", "tracker": "deepsort"},
    "yolo26s_botsort": {"detector": "yolo26s", "tracker": "botsort"},
    "yolo26s_deepsort": {"detector": "yolo26s", "tracker": "deepsort"},
}


def detector_args(detector: str) -> list[str]:
    if detector == "hsv":
        return ["--set", "detector.name=hsv"]
    if detector == "yolo8n":
        return ["--set", "detector.name=yolo", "--set", f"detector.yolo.weights={WEIGHT_8N}",
                "--set", "detector.yolo.imgsz=640"]
    if detector == "yolo26s":
        return ["--set", "detector.name=yolo", "--set", f"detector.yolo.weights={WEIGHT_26S}",
                "--set", "detector.yolo.imgsz=1280"]
    raise ValueError(detector)


def rotate_for(video_name: str) -> int:
    return 1 if "_z_r" in video_name else -1


def run_cmd(cmd: list[str]) -> tuple[int, float]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    return proc.returncode, time.time() - t0


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} | {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def jobs(only: str) -> list[dict]:
    out: list[dict] = []
    vfc_videos = sorted(VIDEOS.rglob("*.mp4"))
    assert len(vfc_videos) == 16, f"ожидалось 16 видео VFC, найдено {len(vfc_videos)}"

    for combo, spec in COMBOS.items():
        if only and only not in combo:
            continue
        for v in vfc_videos:
            stem = v.stem
            if spec["detector"] == "yolo26s" and spec["tracker"] == "botsort":
                # уже прогнано пакетным скриптом
                target = OUT / f"vfc_{stem}"
            else:
                target = MATRIX / combo / stem
            out.append({
                "dataset": "vfc", "combo": combo, "video": v, "target": target,
                "args": ["--set", f"video.rotate90={rotate_for(stem)}",
                         "--set", "trees.method=segments",
                         "--set", f"trees.segments_file={SFRAME}"],
                "gt": None,
            })

    # MinneApple: только недостающие hsv+трекеры (остальные уже в outputs/)
    for combo, spec in COMBOS.items():
        if spec["detector"] != "hsv":
            continue
        if only and only not in combo:
            continue
        out.append({
            "dataset": "minneapple", "combo": combo, "video": MN_VIDEO,
            "target": OUT / f"minneapple_video_hsv_{spec['tracker']}",
            "args": ["--set", "tracker.reid_max_dist=150"],
            "gt": MN_GT,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    todo = jobs(args.only)
    pending = [j for j in todo if not (j["target"] / "summary.json").exists()]
    log(f"всего задач: {len(todo)}, к выполнению: {len(pending)}")
    if args.dry_run:
        for j in pending:
            log(f"  [dry] {j['combo']:16} {j['video'].stem}")
        return 0

    n_done = 0
    for j in pending:
        n_done += 1
        log(f"[{n_done}/{len(pending)}] {j['combo']} / {j['video'].stem}")
        cmd = [sys.executable, "-m", "applecounter", "run",
               "--video", str(j["video"]), "--out", str(j["target"]),
               *detector_args(j["combo"].rsplit("_", 1)[0]),
               "--set", f"tracker.name={j['combo'].rsplit('_', 1)[1]}",
               "--set", "output.save_video=false",
               "--set", "output.save_mosaic=false",
               *j["args"]]
        if j["gt"]:
            cmd += ["--gt", str(j["gt"])]
        code, secs = run_cmd(cmd)
        if code != 0:
            log(f"  ОШИБКА exit={code} ({secs:.0f} c)")
            continue
        sj = json.loads((j["target"] / "summary.json").read_text(encoding="utf-8"))
        log(f"  готово за {secs/60:.1f} мин: яблок {sj.get('total_apples')}, "
            f"треков {sj.get('tracker', {}).get('created_tracks')}, "
            f"двойной счёт {sj.get('tracker', {}).get('n_double_count_events')}")
    log("матрица завершена")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
