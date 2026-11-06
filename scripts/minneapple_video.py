"""Собрать видео из последовательности кадров MinneApple (COCO) и per_frame ground truth.

Пример:
  python scripts/minneapple_video.py ^
      --split data/raw/minneapple/train --sequence 20150919_174730 ^
      --out data/processed/minneapple_video --fps 6
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import cv2


def build(split_dir: Path, sequence: str, out_dir: Path, fps: float, limit: int | None) -> dict:
    data = json.loads((split_dir / "train.json").read_text(encoding="utf-8"))
    images = [im for im in data["images"] if im["file_name"].startswith(f"{sequence}_image")]
    images.sort(key=lambda im: int(re.search(r"_image(\d+)", im["file_name"]).group(1)))
    if limit:
        images = images[:limit]
    if not images:
        raise ValueError(f"последовательность {sequence} не найдена в {split_dir}")

    anns: dict[int, list] = {}
    for a in data.get("annotations", []):
        if a.get("iscrowd"):
            continue
        anns.setdefault(int(a["image_id"]), []).append(a)

    out_dir.mkdir(parents=True, exist_ok=True)
    gt_dir = out_dir / "gt"
    gt_dir.mkdir(exist_ok=True)

    first = cv2.imread(str(split_dir / "images" / images[0]["file_name"]))
    h, w = first.shape[:2]
    video_path = out_dir / f"{sequence}.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    rows = []
    for frame_idx, im in enumerate(images):
        image = cv2.imread(str(split_dir / "images" / im["file_name"]))
        if image is None:
            continue
        writer.write(image)
        for a in anns.get(int(im["id"]), []):
            x, y, bw, bh = (float(v) for v in a["bbox"])
            rows.append({
                "frame": frame_idx,
                "apple_id": int(a["id"]),
                "tree_id": 1,
                "x1": round(x, 2), "y1": round(y, 2),
                "x2": round(x + bw, 2), "y2": round(y + bh, 2),
                "visible_frac": 1.0,
            })
    writer.release()

    per_frame = gt_dir / "per_frame.csv"
    with per_frame.open("w", newline="", encoding="utf-8") as fh:
        writer_csv = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    n_frames = max(r["frame"] for r in rows) + 1
    stats = {
        "video": str(video_path),
        "frames": n_frames,
        "resolution": [w, h],
        "fps": fps,
        "gt_boxes_total": len(rows),
        "mean_apples_per_frame": round(len(rows) / n_frames, 2),
        "per_frame_csv": str(per_frame),
    }
    (out_dir / "manifest.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="data/raw/minneapple/train")
    ap.add_argument("--sequence", default="20150919_174730")
    ap.add_argument("--out", default="data/processed/minneapple_video")
    ap.add_argument("--fps", type=float, default=6.0,
                    help="кадров/с в mp4 (шаг кадров датасета ~5 реальных кадров)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    stats = build(Path(args.split), args.sequence, Path(args.out), args.fps, args.limit)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
