"""Воспроизводимая матрица: общие детекции × внешние BoT-SORT/DeepSORT.

Старые outputs не используются и не перезаписываются. MinneApple-video исключено:
его COCO annotation ID не являются постоянными track ID, а кадры взяты из train.
T1 — отдельный MOT-датасет Zenodo 7384297; VFC — 16 видео и 21 stretch.

Примеры (из корня проекта, в .venv):
  python scripts/run_matrix.py --dry-run
  python scripts/run_matrix.py --dataset vfc --only yolo8n --max-frames 12
  python scripts/run_matrix.py --dataset t1
  python scripts/run_matrix.py --dataset vfc
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "default.yaml"
VFC_ROOT = ROOT / "data" / "raw" / "videofruit" / "Video Fruit Counting"
SFRAME = VFC_ROOT / "stretch_sframe.csv"
VIDEOS = VFC_ROOT / "Videos"
T1 = ROOT / "data" / "raw" / "videofruit" / "T1"
OUT = ROOT / "outputs" / "rework_matrix"
CACHE = ROOT / "outputs" / "rework_cache"
LOG = ROOT / "outputs" / "rework_matrix.log"

WEIGHTS = {
    "yolo8n": (ROOT / "weights/runs/minneapple50/weights/best.pt", 640),
    "yolo26s": (ROOT / "weights/runs/minneapple26s1280/weights/best.pt", 1280),
}
DETECTORS = ("hsv", "yolo8n", "yolo26s")
TRACKERS = ("botsort", "deepsort")


def detector_overrides(name: str) -> list[str]:
    if name == "hsv":
        return ["detector.name=hsv"]
    weights, imgsz = WEIGHTS[name]
    if not weights.is_file():
        raise FileNotFoundError(f"нет весов {name}: {weights}")
    return [
        "detector.name=yolo", f"detector.yolo.weights={weights}",
        f"detector.yolo.imgsz={imgsz}",
    ]


def rotate_for(video_name: str) -> int:
    return 1 if "_z_r" in video_name else -1


def jobs(dataset: str, only: str, max_frames: int | None, frame_stride: int) -> list[dict]:
    result = []
    sources = []
    if dataset in ("vfc", "all"):
        videos = sorted(VIDEOS.rglob("*.mp4"))
        if len(videos) != 16:
            raise ValueError(f"ожидалось 16 видео VFC, найдено {len(videos)}")
        sources.extend(("vfc", path, None) for path in videos)
    if dataset in ("t1", "all"):
        if not (T1 / "img1").is_dir() or not (T1 / "gt" / "gt.txt").is_file():
            raise FileNotFoundError(f"T1 должен содержать img1 и gt/gt.txt: {T1}")
        sources.append(("t1", T1, T1))

    for data_name, source, gt in sources:
        stem = source.stem
        for detector in DETECTORS:
            cache_file = CACHE / data_name / stem / f"{detector}.npz"
            for tracker in TRACKERS:
                label = f"{data_name}/{detector}/{tracker}/{stem}"
                if only and only.lower() not in label.lower():
                    continue
                overrides = [
                    *detector_overrides(detector),
                    f"tracker.name={tracker}",
                    "counting.min_obs=1",
                    "counting.merge_dist=0",
                    "output.save_video=false",
                    "output.save_mosaic=false",
                    "output.save_figures=false",
                    "yield.enabled=false",
                    f"detector.cache_file={cache_file}",
                    f"video.frame_stride={frame_stride}",
                ]
                if max_frames is not None:
                    overrides.append(f"video.max_frames={max_frames}")
                if data_name == "vfc":
                    overrides += [
                        f"video.rotate90={rotate_for(stem)}",
                        "trees.method=segments",
                        f"trees.segments_file={SFRAME}",
                        "trees.assignment=vfc_reference",
                        "trees.subdivide=false",
                    ]
                else:
                    overrides.append("trees.method=none")
                result.append({
                    "dataset": data_name,
                    "detector": detector,
                    "tracker": tracker,
                    "source": source,
                    "gt": gt,
                    "target": OUT / data_name / detector / tracker / stem,
                    "overrides": overrides,
                    "label": label,
                })
    return result


def complete(job: dict) -> bool:
    """Принимать только завершённый прогон с точно совпадающим конфигом."""
    from applecounter.config import load_config

    manifest_path = job["target"] / "run_manifest.json"
    summary_path = job["target"] / "summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected = load_config(str(CONFIG), job["overrides"])
        return (
            manifest.get("status") == "complete"
            and manifest.get("config") == expected.data
            and summary.get("pipeline_version", 0) >= 2
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def cache_file_for_job(job: dict) -> Path | None:
    for override in job["overrides"]:
        if override.startswith("detector.cache_file="):
            return Path(override.split("=", 1)[1])
    return None


def cache_mismatch(cache_path: Path, expected_metadata: dict) -> str | None:
    """Вернуть причину несовместимости кэша; None означает, что reuse безопасен."""
    from applecounter.detect.cache import DetectionCache

    try:
        DetectionCache.load(cache_path, expected_metadata=expected_metadata)
    except (OSError, ValueError) as exc:
        return str(exc)
    return None


def expected_cache_metadata(job: dict, cache: dict) -> dict:
    """Метаданные кэша для job; кэшируются по детекционной части конфигурации."""
    from applecounter.config import load_config
    from applecounter.utils.provenance import detection_cache_metadata

    cache_overrides = tuple(
        override for override in job["overrides"]
        if override.startswith(("detector.", "video.frame_stride", "video.resize",
                                "video.rotate90", "video.max_frames"))
    )
    if cache_overrides not in cache:
        cfg = load_config(str(CONFIG), list(cache_overrides))
        cache[cache_overrides] = detection_cache_metadata(cfg, job["source"])
    return cache[cache_overrides]


def prepare_cache(job: dict, metadata_cache: dict, dry_run: bool = False) -> str | None:
    """Удалить несовместимый кэш только для незавершённой/перезапускаемой задачи."""
    cache_path = cache_file_for_job(job)
    if cache_path is None or not cache_path.exists():
        return None
    reason = cache_mismatch(cache_path, expected_cache_metadata(job, metadata_cache))
    if reason is None:
        return None
    if not dry_run:
        cache_path.unlink(missing_ok=True)
    return f"{cache_path}: {reason}"


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(job: dict) -> tuple[int, float]:
    cmd = [sys.executable, "-m", "applecounter", "run", "--config", str(CONFIG),
           "--video", str(job["source"]), "--out", str(job["target"])]
    if job["gt"] is not None:
        cmd += ["--gt", str(job["gt"])]
    for override in job["overrides"]:
        cmd += ["--set", override]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONIOENCODING"] = "utf-8"
    started = time.time()
    process = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
    return process.returncode, time.time() - started


def collect(jobs_list: list[dict]) -> None:
    rows = []
    for job in jobs_list:
        path = job["target"] / "summary.json"
        if not complete(job):
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        tracking = summary.get("tracking_metrics") or {}
        counting = summary.get("count_metrics") or {}
        rows.append({
            "dataset": job["dataset"], "sequence": job["source"].stem,
            "detector": job["detector"], "tracker": job["tracker"],
            "frames": summary.get("frames"), "tracks": summary.get("total_apples"),
            "count_unit": summary.get("count_unit"),
            "mota": tracking.get("mota"), "idf1": tracking.get("idf1"),
            "idsw": tracking.get("id_switches"),
            "count_mae": counting.get("mae"),
            "total_error_pct": counting.get("total_error_pct"),
            "elapsed_sec": summary.get("elapsed_sec"),
            "run_dir": str(job["target"]),
        })
    if rows:
        OUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(OUT / "matrix_results.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("vfc", "t1", "all"), default="vfc")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", default="", help="подстрока dataset/detector/tracker/video")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--rerun", action="store_true", help="игнорировать валидный manifest")
    args = parser.parse_args()
    if args.frame_stride < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("frame-stride и max-frames должны быть положительными")

    all_jobs = jobs(args.dataset, args.only, args.max_frames, args.frame_stride)
    metadata_cache: dict = {}
    pending: list[dict] = []
    for job in all_jobs:
        if not args.rerun and complete(job):
            continue
        stale = prepare_cache(job, metadata_cache, dry_run=args.dry_run)
        if stale:
            action = "будет пересоздан" if args.dry_run else "пересоздан"
            log(f"несовместимый кэш {action}: {stale}")
        pending.append(job)
    log(f"выбрано {len(all_jobs)}; к выполнению {len(pending)}; старые outputs не используются")
    if args.dry_run:
        for job in pending:
            log(f"[dry] {job['label']} -> {job['target']}")
        return 0

    failures = 0
    for index, job in enumerate(pending, 1):
        log(f"[{index}/{len(pending)}] {job['label']}")
        code, seconds = run(job)
        if code != 0 or not complete(job):
            failures += 1
            log(f"ОШИБКА exit={code}, manifest_valid={complete(job)}, {seconds:.1f} с")
            continue
        summary = json.loads((job["target"] / "summary.json").read_text(encoding="utf-8"))
        cache = summary.get("detection_cache", {})
        log(f"готово: tracks={summary.get('total_apples')}, cache={cache.get('status')}, "
            f"{seconds / 60:.1f} мин")
        collect(all_jobs)
    collect(all_jobs)
    log(f"матрица завершена; ошибок {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
