"""Экспорт результатов прогона в формат VideoFruitCounting (Felip-Pomés et al., 2026).

Репозиторий: https://github.com/imatge-upc/VideoFruitCounting

Их конвейер: YOLOv5 -> ByteTrack -> `all_predictions.csv` (пофреймовые боксы
треков) -> `assign_apples.py` распределяет яблока по stretch'ам по SFRAME-
границам (кадры, в которые лента-маркер оказывается по центру кадра).

Формат `all_predictions.csv` (без учёта заголовка, читается их read_csv):
    frame, track_id, x, y, w, h, score, stretch

Наш мост: `applecounter run` сохраняет `tracks_frame.csv` (экран-боксы
подтверждённых треков), а `applecounter vfc-export` собирает из него
`all_predictions.csv`. Колонка stretch берётся:
  * из SFRAME/segments-файла (кадровые границы, их формат
    `<video>,f0,f1,...,fN` уже понимает наш load_segments);
  * иначе — из tree_id в tracks.csv (привязка дерева, уже посчитанная
    нашим пайплайном).

После экспорта их скрипт назначения stretch запускается так:
    python external/VideoFruitCounting/assign_apples.py \
        --data_path <каталог с подкаталогами-видео> --sframe_file <SFRAME.csv>
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from ..trees.segment import TreeMap, load_segments

VFC_COLUMNS = ["frame", "track_id", "x", "y", "w", "h", "score", "stretch"]


def export_vfc(run_dir: str, out: Optional[str] = None,
               segments_file: Optional[str] = None,
               video_name: Optional[str] = None) -> Path:
    """Собрать `all_predictions.csv` из прогона `applecounter run`.

    run_dir       — каталог прогона (нужен tracks_frame.csv);
    out           — путь файла результата; по умолчанию
                    <run>/vfc/<видео>/all_predictions.csv — структура,
                    которую принимает их assign_apples.py (--data_path <run>/vfc);
    segments_file — SFRAME/segments CSV для колонки stretch;
    video_name    — имя видео (строка в SFRAME); по умолчанию берётся
                    из summary.json.
    """
    run = Path(run_dir)
    frames_path = run / "tracks_frame.csv"
    if not frames_path.exists():
        raise FileNotFoundError(
            f"в {run} нет tracks_frame.csv — сначала выполните `applecounter run` "
            "(файл пишется автоматически при прогоне видео)"
        )
    df = pd.read_csv(frames_path)
    if df.empty:
        raise ValueError(f"{frames_path} пуст — треки не сохранялись")

    # имя видео для выбора строки SFRAME и структуры каталогов
    if video_name is None:
        summary_path = run / "summary.json"
        if summary_path.exists():
            try:
                video_name = json.loads(summary_path.read_text(encoding="utf-8")).get("video")
            except (json.JSONDecodeError, OSError):
                video_name = None

    # stretch: по кадровым границам (SFRAME) либо по уже посчитанному tree_id
    if segments_file:
        segments = load_segments(segments_file, video_name=video_name)
        tree_map = TreeMap(method="segments", frame_segments=segments)
        stretch = df["frame"].map(lambda f: tree_map.assign_frames(int(f)))
    elif (run / "tracks.csv").exists():
        tracks = pd.read_csv(run / "tracks.csv")[["track_id", "tree_id"]]
        df = df.merge(tracks, on="track_id", how="left")
        stretch = df["tree_id"].fillna(0).astype(int)
    else:
        stretch = pd.Series(0, index=df.index, dtype=int)

    out_df = pd.DataFrame({
        "frame": df["frame"].astype(int),
        "track_id": df["track_id"].astype(int),
        "x": df["x1"].astype(float),
        "y": df["y1"].astype(float),
        "w": (df["x2"] - df["x1"]).astype(float),
        "h": (df["y2"] - df["y1"]).astype(float),
        "score": df["conf"].astype(float),
        "stretch": stretch.astype(int),
    })[VFC_COLUMNS]

    if out is not None:
        out_path = Path(out)
    else:
        stem = Path(video_name).stem if video_name else run.name
        out_path = run / "vfc" / stem / "all_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return out_path
