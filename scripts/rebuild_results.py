"""Пересобрать outputs/rework_matrix/matrix_results.csv из run_manifest.json на диске.

Идемпотентен: каждый прогон оставляет в своём каталоге run_manifest.json +
summary.json, поэтому сводную таблицу можно восстановить в любой момент,
даже если очередная фаза её перезаписала.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MATRIX_OUT = ROOT / "outputs" / "rework_matrix"
RESULTS = MATRIX_OUT / "matrix_results.csv"


def main() -> int:
    rows = []
    for manifest_path in sorted(MATRIX_OUT.glob("*/*/*/*/run_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            continue
        rel = manifest_path.relative_to(MATRIX_OUT).parts
        summary_path = manifest_path.parent / "summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        tracking = summary.get("tracking_metrics") or {}
        counting = summary.get("count_metrics") or {}
        rows.append({
            "dataset": rel[0], "sequence": rel[3],
            "detector": rel[1], "tracker": rel[2],
            "frames": summary.get("frames"),
            "tracks": summary.get("total_apples"),
            "count_unit": summary.get("count_unit"),
            "mota": tracking.get("mota"), "idf1": tracking.get("idf1"),
            "idsw": tracking.get("id_switches"),
            "count_mae": counting.get("mae"),
            "total_error_pct": counting.get("total_error_pct"),
            "elapsed_sec": summary.get("elapsed_sec"),
            "run_dir": str(manifest_path.parent),
        })
    if not rows:
        print("не найдено завершённых прогонов", file=sys.stderr)
        return 1
    frame = pd.DataFrame(rows).sort_values(
        ["dataset", "sequence", "detector", "tracker"])
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS, index=False)
    print(f"пересобрано строк: {len(frame)} -> {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
