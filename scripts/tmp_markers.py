"""Извлечь кадры на границах stretch (SFRAME) и приблизить центр — маркер."""
import csv
import cv2
from pathlib import Path

ROOT = Path(r"G:\opencode\soft\applecounter3")
VFC = ROOT / "data" / "raw" / "videofruit" / "Video Fruit Counting"
OUT = ROOT / "outputs" / "tmp_markers"
OUT.mkdir(parents=True, exist_ok=True)

sframe = {}
for row in csv.reader((VFC / "stretch_sframe.csv").open(encoding="utf-8")):
    if row:
        sframe[row[0].strip()] = [int(x) for x in row[1:]]

videos = sorted((VFC / "Videos").rglob("*.mp4"))
for src in videos[:3]:
    stem = src.stem
    bounds = sframe.get(stem)
    if not bounds:
        print(stem, "-> нет в SFRAME")
        continue
    cap = cv2.VideoCapture(str(src))
    for bi in (1, 5, 10):
        pos = bounds[bi]
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok:
            continue
        rot = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE if "_z_r" in stem
                         else cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w = rot.shape[:2]
        full = cv2.resize(rot, (w // 3, h // 3))
        cv2.imwrite(str(OUT / f"{stem[:20]}_b{bi}_full.png"), full)
        # центральная зона (маркер по центру кадра по их описанию)
        cx0, cy0 = int(w * 0.30), int(h * 0.30)
        crop = rot[cy0:int(h * 0.70), cx0:int(w * 0.70)]
        crop = cv2.resize(crop, (crop.shape[1] // 2, crop.shape[0] // 2))
        cv2.imwrite(str(OUT / f"{stem[:20]}_b{bi}_crop.png"), crop)
        print("saved", stem[:20], "boundary", bi, "frame", pos)
    cap.release()
print("готово:", OUT)
