"""Генерация иллюстраций для промежуточного отчёта (MinneApple + T1)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(ROOT / "src"))

from applecounter.config import Config  # noqa: E402
from applecounter.detect.base import build_detector  # noqa: E402
from applecounter.viz.draw import color_for_id  # noqa: E402

GT_COLOR = (40, 200, 60)
PRED_COLOR = (0, 220, 255)

YOLO8N = {"name": "yolo", "yolo": {"weights": str(ROOT / "weights/runs/minneapple50/weights/best.pt"),
                                  "imgsz": 640, "conf": 0.25}}
YOLO26S = {"name": "yolo", "yolo": {"weights": str(ROOT / "weights/runs/minneapple26s1280/weights/best.pt"),
                                   "imgsz": 1280, "conf": 0.25}}


def draw_boxes(img, boxes, color, thickness=2):
    out = img.copy()
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b[:4]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out


# ---------------------------------------------------------------- MinneApple

def yolo_label_to_xyxy(txt: Path, w: int, h: int) -> np.ndarray:
    boxes = []
    for line in txt.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = map(float, parts[:5])
        boxes.append([ (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h ])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def minneapple_figures() -> None:
    val_dir = ROOT / "data/processed/minneapple/images/val"
    lab_dir = ROOT / "data/processed/minneapple/labels/val"
    # изображение с наибольшим числом GT
    best, best_n = None, -1
    for txt in sorted(lab_dir.glob("*.txt")):
        n = len(txt.read_text().splitlines())
        if n > best_n:
            best, best_n = txt, n
    img_path = val_dir / (best.stem + ".png")
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    gt = yolo_label_to_xyxy(best, w, h)
    print(f"minneapple: {best.stem}, GT boxes={len(gt)}")

    detections = {}
    for tag, det_cfg in [("hsv", {"name": "hsv"}), ("yolo8n", YOLO8N), ("yolo26s", YOLO26S)]:
        cfg = Config({"detector": det_cfg, "project": {"device": "cuda:0"}})
        det = build_detector(cfg)
        d = det.detect(img)
        det.close()
        detections[tag] = d
        out = draw_boxes(img, gt, GT_COLOR, 1)
        out = draw_boxes(out, d[:, :4], PRED_COLOR, 2)
        cv2.putText(out, f"{tag}  GT={len(gt)} (green)  pred={len(d)} (yellow)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(FIG / f"minneapple_{tag}.png"), out)

    # метрики детекции по bench-прогонам
    rows = []
    for tag in ["hsv", "yolo8n_640", "yolo26s_1280"]:
        bench = json.loads((ROOT / f"outputs/minneapple_detection/bench_{tag}.json").read_text(encoding="utf-8"))
        key = "hsv" if tag == "hsv" else "yolo"
        m = bench[key]
        rows.append({"detector": tag, "ap50": m["ap50"], "precision": m["precision"],
                     "recall": m["recall"], "f1": m["f1"],
                     "count_mae": m["count_mae_per_image"]})
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(df))
    for i, col in enumerate(["ap50", "precision", "recall", "f1"]):
        axes[0].bar(x + (i - 1.5) * 0.2, df[col], width=0.2, label=col)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df.detector)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("MinneApple val (331 img): detection metrics")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(df.detector, df.count_mae, color=["#c44", "#48c", "#4a4"])
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_title("Count MAE per image (GT mean = 37.1)")
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "minneapple_metrics.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- T1

def t1_figures() -> None:
    img_dir = ROOT / "data/raw/videofruit/T1/img1"
    frames = [216, 217, 220]  # кадры с максимумом активных треков (45)
    runs = [("yolo8n", "botsort"), ("yolo26s", "botsort"), ("yolo8n", "deepsort"), ("hsv", "botsort")]
    for det, tr in runs:
        tf = pd.read_csv(ROOT / f"outputs/rework_matrix/t1/{det}/{tr}/T1/tracks_frame.csv")
        panels = []
        for f in frames:
            img = cv2.imread(str(img_dir / f"{f+1:06d}.PNG"))
            if img is None:
                continue
            sub = tf[tf.frame == f]
            for _, r in sub.iterrows():
                color = color_for_id(int(r.track_id))
                cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), color, 2)
                cv2.putText(img, str(int(r.track_id)), (int(r.x1), max(12, int(r.y1) - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            cv2.putText(img, f"frame {f} | tracks: {len(sub)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            panels.append(img)
        if panels:
            h_max = max(p.shape[0] for p in panels)
            panels = [cv2.copyMakeBorder(p, 0, h_max - p.shape[0], 0, 0, cv2.BORDER_CONSTANT) for p in panels]
            row = np.hstack(panels)
            cv2.imwrite(str(FIG / f"t1_{det}_{tr}.png"), row)
            print(f"t1 {det}/{tr}: saved, tracks on frames {[len(tf[tf.frame==f]) for f in frames]}")

    # сводные графики метрик
    rows = []
    for det in ["hsv", "yolo8n", "yolo26s"]:
        for tr in ["botsort", "deepsort"]:
            s = json.loads((ROOT / f"outputs/rework_matrix/t1/{det}/{tr}/T1/summary.json").read_text(encoding="utf-8"))
            rows.append({"det": det, "tr": tr,
                         "mota": s["tracking_metrics"]["mota"],
                         "idf1": s["tracking_metrics"]["idf1"],
                         "idsw": s["tracking_metrics"]["id_switches"],
                         "ap50": s["detection_metrics"]["ap50"],
                         "n_tracks": s["total_apples"]})
    df = pd.DataFrame(rows)
    labels = [f"{d}+{t}" for d, t in zip(df.det, df.tr)]
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(x, df.mota, color=["#48c" if t == "botsort" else "#c84" for t in df.tr])
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_title("MOTA (выше = лучше)")
    axes[1].bar(x, df.idf1, color=["#48c" if t == "botsort" else "#c84" for t in df.tr])
    axes[1].set_title("IDF1 (выше = лучше)")
    axes[1].set_ylim(0, 0.6)
    axes[2].bar(x, df.idsw, color=["#48c" if t == "botsort" else "#c84" for t in df.tr])
    axes[2].set_title("ID switches (ниже = лучше)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("T1 (1290 кадров, GT: 939 яблок): трекинг-метрики")
    fig.tight_layout()
    fig.savefig(FIG / "t1_metrics.png", dpi=140)
    plt.close(fig)

    # распределение длин треков для лучшей комбинации против GT
    fig, ax = plt.subplots(figsize=(7, 4))
    for det, tr, lab in [("yolo8n", "botsort", "yolo8n+botsort"), ("hsv", "botsort", "hsv+botsort (baseline)")]:
        t = pd.read_csv(ROOT / f"outputs/rework_matrix/t1/{det}/{tr}/T1/tracks.csv")
        ax.hist(t.n_obs, bins=50, alpha=0.55, label=f"{lab} (n={len(t)})")
    ax.set_xlabel("число кадров наблюдения трека")
    ax.set_ylabel("число треков")
    ax.set_title("T1: распределение длин треков (GT яблок = 939)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "t1_track_lengths.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    minneapple_figures()
    t1_figures()
    print("figures ->", FIG)
