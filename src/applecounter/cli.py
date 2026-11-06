"""Консольный интерфейс: python -m applecounter <команда> / applecounter <команда>."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import Config, load_config
from .utils.logging import get_logger, setup_logging

logger = get_logger()


# ---------------------------------------------------------------------------
# Общие аргументы
# ---------------------------------------------------------------------------
def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="путь к YAML-конфигурации")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="оверрайд конфигурации, напр. tracker.n_init=3 (можно повторять)")
    parser.add_argument("-v", "--verbose", action="store_true", help="подробный вывод логов")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="applecounter",
        description="Подсчёт яблок на видео пролёта дрона: детекция, трекинг без повторов, "
                    "карта деревьев, прогноз урожая, отчёт.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # synth ---------------------------------------------------------------
    p = sub.add_parser("synth", help="сгенерировать синтетическое видео с ground truth")
    p.add_argument("--out", default="data/synthetic", help="каталог вывода")
    p.add_argument("--trees", type=int, default=6)
    p.add_argument("--apples-min", type=int, default=18)
    p.add_argument("--apples-max", type=int, default=30)
    p.add_argument("--frames", type=int, default=None, help="явное число кадров")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--distractors", type=int, default=0)
    _add_common(p)
    p.set_defaults(func=cmd_synth)

    # download ------------------------------------------------------------
    p = sub.add_parser("download", help="скачать общеупотребимые датасеты")
    p.add_argument("--dataset", default=None, help="minneapple | acfr | videofruit")
    p.add_argument("--out", default="data/raw", help="каталог загрузки")
    p.add_argument("--url", default=None, help="прямая ссылка (если известна)")
    p.add_argument("--archive", default=None, help="уже скачанный архив (ручной режим)")
    p.add_argument("--list", action="store_true", help="показать список датасетов")
    _add_common(p)
    p.set_defaults(func=cmd_download)

    # prepare -------------------------------------------------------------
    p = sub.add_parser("prepare", help="конвертировать датасет в формат YOLO")
    p.add_argument("--dataset", required=True, help="minneapple | acfr")
    p.add_argument("--src", required=True, help="каталог распакованного датасета")
    p.add_argument("--out", default=None, help="каталог выхода (по умолчанию data/processed/<dataset>)")
    p.add_argument("--val-ratio", type=float, default=0.15)
    _add_common(p)
    p.set_defaults(func=cmd_prepare)

    # train ---------------------------------------------------------------
    p = sub.add_parser("train", help="дообучить YOLO на подготовленном датасете")
    p.add_argument("--data", required=True, help="путь к data.yaml")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out", default="weights")
    p.add_argument("--name", default=None)
    _add_common(p)
    p.set_defaults(func=cmd_train)

    # bench ---------------------------------------------------------------
    p = sub.add_parser("bench", help="бенчмарк детекторов (hsv|yolo) на val-сплите датасета")
    p.add_argument("--data", required=True, help="каталог YOLO-датасета (есть data.yaml)")
    p.add_argument("--detector", default="both", help="hsv | yolo | both")
    p.add_argument("--yolo-weights", default=None, help="веса YOLO (иначе из конфига detector.yolo.weights)")
    p.add_argument("--limit", type=int, default=None, help="ограничить число изображений")
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--out", default=None, help="куда записать bench.json (по умолчанию <data>/bench.json)")
    _add_common(p)
    p.set_defaults(func=cmd_bench)

    # run -----------------------------------------------------------------
    p = sub.add_parser("run", help="полный проход по видео: детекция -> трекинг -> счёт -> отчёт")
    p.add_argument("--video", required=True)
    p.add_argument("--gt", default=None, help="каталог ground truth (gt_counts.csv, per_frame.csv)")
    p.add_argument("--out", default=None, help="каталог результатов (по умолчанию outputs/<имя видео>)")
    _add_common(p)
    p.set_defaults(func=cmd_run)

    # evaluate ------------------------------------------------------------
    p = sub.add_parser("evaluate", help="оценить готовый прогон по ground truth")
    p.add_argument("--run", required=True, help="каталог прогона (с counts.csv)")
    p.add_argument("--gt", required=True)
    p.add_argument("--out", default=None, help="куда записать eval.json (по умолчанию в --run)")
    _add_common(p)
    p.set_defaults(func=cmd_evaluate)

    # yield ---------------------------------------------------------------
    p = sub.add_parser("yield", help="прогноз урожая по counts.csv")
    p.add_argument("--counts", required=True, help="counts.csv из прогона")
    p.add_argument("--out", default=None, help="каталог выхода (по умолчанию рядом со counts.csv)")
    p.add_argument("--regression-csv", default=None, help="полевой контроль (tree_id,count,mass_kg)")
    _add_common(p)
    p.set_defaults(func=cmd_yield)

    # report --------------------------------------------------------------
    p = sub.add_parser("report", help="пересобрать отчёт и графики из сохранённого прогона")
    p.add_argument("--run", required=True)
    _add_common(p)
    p.set_defaults(func=cmd_report)

    # retree --------------------------------------------------------------
    p = sub.add_parser("retree", help="пересчитать привязку яблок к деревьям по сохранённым "
                                      "артефактам прогона (без перезапуска видео и GPU)")
    p.add_argument("--run", required=True, help="каталог прогона (tracks.csv, tracks_frame.csv)")
    p.add_argument("--segments", default=None,
                   help="SFRAME/segments CSV (по умолчанию — из summary.json tree_meta.file)")
    p.add_argument("--mosaic", action="store_true",
                   help="также перегенерировать мозаику пролёта (нужен rot.txt или --set video.rotate90)")
    _add_common(p)
    p.set_defaults(func=cmd_retree)

    # vfc-export ----------------------------------------------------------
    p = sub.add_parser("vfc-export", help="экспорт прогона в формат VideoFruitCounting "
                                          "(all_predictions.csv для assign_apples.py)")
    p.add_argument("--run", required=True, help="каталог прогона (tracks_frame.csv)")
    p.add_argument("--segments", default=None,
                   help="SFRAME/segments CSV для колонки stretch (иначе tree_id из tracks.csv)")
    p.add_argument("--video-name", default=None,
                   help="имя видео для выбора строки SFRAME (по умолчанию из summary.json)")
    p.add_argument("--out", default=None,
                   help="файл результата (по умолчанию <run>/vfc/<видео>/all_predictions.csv)")
    _add_common(p)
    p.set_defaults(func=cmd_vfc_export)

    # demo ----------------------------------------------------------------
    p = sub.add_parser("demo", help="демо end-to-end: синтетика -> подсчёт -> метрики -> отчёт")
    p.add_argument("--out", default="outputs/demo")
    p.add_argument("--frames", type=int, default=None,
                   help="число кадров синтетики (по умолчанию — автоматически, на весь пролёт)")
    p.add_argument("--trees", type=int, default=6)
    p.add_argument("--force", action="store_true", help="перегенерировать синтетику")
    _add_common(p)
    p.set_defaults(func=cmd_demo)

    # journals ------------------------------------------------------------
    p = sub.add_parser("journals", help="целевые журналы ВАК для статьи")
    p.add_argument("--category", default=None, help="К1 | К2 | К3")
    _add_common(p)
    p.set_defaults(func=cmd_journals)

    return parser


# ---------------------------------------------------------------------------
# Команды
# ---------------------------------------------------------------------------
def cmd_synth(args: argparse.Namespace, cfg: Config) -> int:
    from .data.synthetic import SyntheticParams, generate_synthetic

    params = SyntheticParams(
        width=args.width,
        height=args.height,
        n_trees=args.trees,
        apples_min=args.apples_min,
        apples_max=args.apples_max,
        frames=args.frames,
        seed=args.seed,
        distractors=args.distractors,
    )
    manifest = generate_synthetic(args.out, params)
    print(f"видео: {manifest['video']}")
    print(f"кадров: {manifest['n_frames']}, деревьев: {manifest['n_trees']}, "
          f"яблок: {manifest['n_apples']}")
    print(f"ground truth: {manifest['gt']['counts']}")
    return 0


def cmd_download(args: argparse.Namespace, cfg: Config) -> int:
    from .data.download import download_dataset, list_datasets

    if args.list or not args.dataset:
        print(list_datasets())
        return 0 if args.list else 2
    result = download_dataset(args.dataset, args.out, url=args.url, archive=args.archive)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("downloaded"):
        print(result.get("notes", ""))
        return 1
    return 0


def cmd_prepare(args: argparse.Namespace, cfg: Config) -> int:
    from .data.convert import convert_dataset

    out = args.out or str(Path("data/processed") / args.dataset)
    result = convert_dataset(args.dataset, args.src, out, val_ratio=args.val_ratio)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_train(args: argparse.Namespace, cfg: Config) -> int:
    from .detect.yolo import train_yolo

    best = train_yolo(cfg, args.data, epochs=args.epochs, model=args.model,
                      out_dir=args.out, batch=args.batch, imgsz=args.imgsz, name=args.name)
    print(f"лучшие веса: {best}")
    print(f"подключить:  --set detector.name=yolo --set detector.yolo.weights={best}")
    return 0


def cmd_bench(args: argparse.Namespace, cfg: Config) -> int:
    import cv2
    import yaml

    from .detect.base import build_detector
    from .eval.metrics import detection_metrics

    data_dir = Path(args.data)
    yaml_path = data_dir / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml не найден: {yaml_path}")
    meta = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root = Path(str(meta.get("path", data_dir)))
    val_dir = root / str(meta.get("val", "images/val"))
    lbl_dir = root / "labels" / val_dir.name

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(p for p in val_dir.iterdir() if p.suffix.lower() in exts)
    if args.limit:
        images = images[: int(args.limit)]
    if not images:
        raise ValueError(f"val-сплит пуст: {val_dir}")

    def load_gt(img_path: Path) -> np.ndarray:
        boxes = []
        lbl = lbl_dir / f"{img_path.stem}.txt"
        if lbl.exists():
            im = cv2.imread(str(img_path))
            h, w = im.shape[:2]
            for line in lbl.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
                boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h,
                              (cx + bw / 2) * w, (cy + bh / 2) * h, 1.0, 0.0))
        return np.asarray(boxes, dtype=np.float32).reshape(-1, 6)

    gts = [load_gt(p) for p in images]
    names = ["hsv", "yolo"] if args.detector == "both" else [args.detector.lower()]
    results: dict = {"data": str(data_dir), "n_images": len(images), "iou_thr": args.iou_thr}
    for name in names:
        cfg.set("detector.name", name)
        if name == "yolo" and args.yolo_weights:
            cfg.set("detector.yolo.weights", str(args.yolo_weights))
        detector = build_detector(cfg)
        preds, count_err = [], []
        for i, img_path in enumerate(images):
            image = cv2.imread(str(img_path))
            det = detector.detect(image)
            preds.append(det)
            count_err.append(abs(len(det) - len(gts[i])))
        detector.close()
        m = detection_metrics(preds, gts, args.iou_thr)
        m["count_mae_per_image"] = float(np.mean(count_err))
        m["mean_pred"] = float(np.mean([len(p) for p in preds]))
        m["mean_gt"] = float(np.mean([len(g) for g in gts]))
        results[name] = m
        logger.info("%s: AP@%.2f=%.3f P=%.3f R=%.3f F1=%.3f count-MAE=%.2f",
                    name, args.iou_thr, m["ap50"], m["precision"], m["recall"], m["f1"],
                    m["count_mae_per_image"])

    out_path = Path(args.out) if args.out else data_dir / "bench.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{'детектор':<8} {'AP@50':>7} {'P':>6} {'R':>6} {'F1':>6} {'count-MAE':>9}")
    for name in names:
        m = results[name]
        print(f"{name:<8} {m['ap50']:7.3f} {m['precision']:6.3f} {m['recall']:6.3f} "
              f"{m['f1']:6.3f} {m['count_mae_per_image']:9.2f}")
    print(f"записано: {out_path}")
    return 0


def cmd_run(args: argparse.Namespace, cfg: Config) -> int:
    from .pipeline import run_video

    out = args.out or str(Path(cfg.get("paths.runs_dir", "outputs")) / Path(args.video).stem)
    summary = run_video(cfg, args.video, out, gt_dir=args.gt)
    _print_summary(summary)
    print(f"\nотчёт: {summary.get('paths', {}).get('report')}")
    return 0


def cmd_evaluate(args: argparse.Namespace, cfg: Config) -> int:
    from .pipeline import evaluate_run

    out = args.out or args.run
    result = evaluate_run(args.run, args.gt, cfg)
    cm = result.get("count_metrics")
    if cm:
        print(f"MAE={cm['mae']:.2f}  RMSE={cm['rmse']:.2f}  "
              f"ошибка итога={cm['total_error_pct']:.1f}%  "
              f"в допуске {cm['within_tolerance']}/{cm['n_trees']}")
    print(f"записано: {Path(out) / 'eval.json'}")
    return 0


def cmd_yield(args: argparse.Namespace, cfg: Config) -> int:
    from .yieldpred.model import forecast_counts

    counts_path = Path(args.counts)
    df = pd.read_csv(counts_path)
    per_tree = {int(row["tree_id"]): int(row["count"])
                for _, row in df.iterrows() if int(row["tree_id"]) > 0}
    if args.regression_csv:
        cfg.set("yield.method", "regression")
        cfg.set("yield.regression_csv", args.regression_csv)
    forecast = forecast_counts(per_tree, cfg)
    out_dir = Path(args.out) if args.out else counts_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(forecast.as_rows()).to_csv(out_dir / "yield.csv", index=False)
    (out_dir / "yield.json").write_text(
        json.dumps(
            {
                "method": forecast.method,
                "total_kg": forecast.total_kg,
                "total_sigma": forecast.total_sigma,
                "ci": [forecast.ci_lo, forecast.ci_hi],
                "params": forecast.params,
                "per_tree": forecast.as_rows(),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"прогноз ({forecast.method}): {forecast.total_kg:.1f} кг "
          f"[{forecast.ci_lo:.1f} … {forecast.ci_hi:.1f}] -> {out_dir / 'yield.json'}")
    return 0


def cmd_report(args: argparse.Namespace, cfg: Config) -> int:
    """Пересобрать отчёт/графики из сохранённых артефактов прогона."""
    from .counting.counter import CountResult, TrackRecord
    from .viz import report as viz_report

    run = Path(args.run)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    tracks = pd.read_csv(run / "tracks.csv")
    records = [
        TrackRecord(
            track_id=int(row["track_id"]),
            first_frame=int(row["first_frame"]),
            last_frame=int(row["last_frame"]),
            n_obs=int(row["n_obs"]),
            world_center=np.array([row["x"], row["y"]], dtype=float),
            tree_id=int(row["tree_id"]),
            mean_conf=float(row["mean_conf"]),
        )
        for _, row in tracks.iterrows()
    ]
    counts_df = pd.read_csv(run / "counts.csv")
    per_tree = {int(row["tree_id"]): int(row["count"]) for _, row in counts_df.iterrows()}
    per_frame_ids: List[int] = []
    if (run / "per_frame.csv").exists():
        per_frame_ids = pd.read_csv(run / "per_frame.csv")["unique_counted"].astype(int).tolist()

    count_result = CountResult(
        total=len(records),
        per_tree=per_tree,
        unassigned=int(summary.get("unassigned", 0)),
        records=records,
        per_frame_ids=per_frame_ids,
    )

    figures_dir = run / "figures"
    figures_dir.mkdir(exist_ok=True)
    figures = {}
    figures["Подсчёт по деревьям"] = viz_report.fig_counts_per_tree(
        count_result, None, figures_dir / "counts_per_tree.png").name
    if per_frame_ids:
        figures["Накопленный счёт"] = viz_report.fig_cumulative(
            per_frame_ids, figures_dir / "cumulative_count.png").name
    figures["Структура метода"] = viz_report.fig_pipeline(figures_dir / "pipeline.png").name
    if summary.get("count_metrics"):
        figures["Ошибка по деревьям"] = viz_report.fig_error_per_tree(
            summary["count_metrics"]["per_tree"], figures_dir / "error_per_tree.png").name

    path = viz_report.write_report(
        run, summary, count_result,
        det_metrics=summary.get("detection_metrics"),
        count_metrics=summary.get("count_metrics"),
        track_metrics=summary.get("tracking_metrics"),
        figures={k: f"figures/{v}" for k, v in figures.items()},
    )
    print(f"отчёт: {path}")
    return 0


def cmd_retree(args: argparse.Namespace, cfg: Config) -> int:
    """Пересчитать привязку яблок к деревьям по сохранённым артефактам.

    Не читает видео и не использует GPU: мировые боксы восстанавливаются из
    tracks_frame.csv (экранные) минус смещения камеры из per_frame.csv
    (стабилизация — чистый сдвиг), карта деревьев строится заново
    (в т.ч. subdivide на отдельные деревья), обновляются tracks.csv,
    counts.csv, summary.json, yield.* и пересобирается отчёт.
    """
    from .counting.counter import TrackRecord
    from .trees.segment import build_tree_map
    from .yieldpred.model import forecast_counts

    run = Path(args.run)
    summary_path = run / "summary.json"
    tracks_path = run / "tracks.csv"
    frames_path = run / "tracks_frame.csv"
    per_frame_path = run / "per_frame.csv"
    for p in (summary_path, tracks_path, frames_path, per_frame_path):
        if not p.exists():
            raise FileNotFoundError(f"нет {p} — retree выполняется только для полного прогона")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    tracks = pd.read_csv(tracks_path)
    tf = pd.read_csv(frames_path)
    pf = pd.read_csv(per_frame_path)

    video_name = Path(str(summary.get("video", ""))).name
    tree_meta0 = summary.get("tree_meta") or {}
    segments_file = args.segments or tree_meta0.get("file")
    if str(summary.get("tree_method", "")).startswith("segments") or segments_file:
        cfg.set("trees.method", "segments")
        if segments_file:
            cfg.set("trees.segments_file", str(segments_file))

    # мировые боксы детекций: экранные минус смещение камеры
    offs = pf.set_index("frame")[["cam_dx", "cam_dy"]]
    tf = tf.join(offs, on="frame")
    if tf["cam_dx"].isna().any():
        raise ValueError("per_frame.csv не покрывает кадры tracks_frame.csv")
    det = np.column_stack([
        tf["x1"].to_numpy(float) - tf["cam_dx"].to_numpy(float),
        tf["y1"].to_numpy(float) - tf["cam_dy"].to_numpy(float),
        tf["x2"].to_numpy(float) - tf["cam_dx"].to_numpy(float),
        tf["y2"].to_numpy(float) - tf["cam_dy"].to_numpy(float),
        tf["frame"].to_numpy(float),
    ])
    points = np.column_stack([(det[:, 0] + det[:, 2]) * 0.5, (det[:, 1] + det[:, 3]) * 0.5])
    tree_map = build_tree_map(points, cfg, video_name=video_name, detections=det)

    # переназначение треков: мировой центр трека (для sub_edges — по оси ряда)
    centers = tracks[["x", "y"]].to_numpy(float)
    mid = (tracks["first_frame"].to_numpy(int) + tracks["last_frame"].to_numpy(int)) // 2
    tracks["tree_id"] = [
        int(tree_map.assign_track(centers[i:i + 1], [int(mid[i])]))
        for i in range(len(tracks))
    ]
    tracks.to_csv(tracks_path, index=False)

    vc = tracks["tree_id"].value_counts()
    per_tree = {int(k): int(v) for k, v in sorted(vc.items()) if int(k) > 0}
    unassigned = int((tracks["tree_id"] <= 0).sum())
    pd.DataFrame([{"tree_id": k, "count": v} for k, v in per_tree.items()],
                 columns=["tree_id", "count"]).to_csv(run / "counts.csv", index=False)

    summary["n_trees"] = tree_map.n_trees
    summary["tree_method"] = tree_map.method
    summary["tree_meta"] = {
        **tree_map.meta,
        "boundaries": [float(b) for b in np.atleast_1d(tree_map.boundaries)],
        "origin": [float(v) for v in tree_map.origin],
        "direction": [float(v) for v in tree_map.direction],
        "sub_edges": [float(b) for b in np.atleast_1d(getattr(tree_map, "sub_edges", np.zeros(0)))],
    }
    summary["counts"] = {str(k): v for k, v in per_tree.items()}
    summary["unassigned"] = unassigned

    # прогноз урожая по новой разбивке (total_apples не меняется)
    try:
        forecast = forecast_counts(per_tree, cfg)
        summary["yield"] = {
            "method": forecast.method,
            "total_kg": round(forecast.total_kg, 2),
            "sigma_kg": round(forecast.total_sigma, 2),
            "ci_lo": round(forecast.ci_lo, 2),
            "ci_hi": round(forecast.ci_hi, 2),
            "params": forecast.params,
        }
        pd.DataFrame(forecast.as_rows()).to_csv(run / "yield.csv", index=False)
        (run / "yield.json").write_text(
            json.dumps(
                {
                    "method": forecast.method,
                    "total_kg": forecast.total_kg,
                    "total_sigma": forecast.total_sigma,
                    "ci": [forecast.ci_lo, forecast.ci_hi],
                    "params": forecast.params,
                    "per_tree": forecast.as_rows(),
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("прогноз урожая не рассчитан: %s", exc)
        summary["yield"] = None

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                            encoding="utf-8")

    # мозаика (по желанию): поворот исходного прогона берётся из rot.txt
    if args.mosaic:
        from .viz.report import make_mosaic_figure

        rot_file = run / "rot.txt"
        if rot_file.exists():
            cfg.set("video.rotate90", int(rot_file.read_text(encoding="ascii").strip() or 0))
        records = [
            TrackRecord(
                track_id=int(row["track_id"]),
                first_frame=int(row["first_frame"]),
                last_frame=int(row["last_frame"]),
                n_obs=int(row["n_obs"]),
                world_center=np.array([row["x"], row["y"]], dtype=float),
                tree_id=int(row["tree_id"]),
                mean_conf=float(row["mean_conf"]),
            )
            for _, row in tracks.iterrows()
        ]
        make_mosaic_figure(
            str(summary.get("video")),
            pf[["cam_dx", "cam_dy"]].to_numpy(float),
            cfg,
            records,
            run / "mosaic.png",
            tree_map=tree_map,
            stride=int(cfg.get("output.mosaic_stride", 4)),
            scale=float(cfg.get("output.mosaic_scale", 0.5)),
        )
        print(f"мозаика: {run / 'mosaic.png'}")

    cmd_report(args, cfg)
    print(f"деревьев: {tree_map.n_trees} | без привязки: {unassigned} | "
          f"уникальных яблок: {summary.get('total_apples')}")
    return 0


def cmd_vfc_export(args: argparse.Namespace, cfg: Config) -> int:
    """Мост к пайплайну VideoFruitCounting: экспорт в их формат all_predictions.csv."""
    from .data.vfc import export_vfc

    out_path = export_vfc(args.run, out=args.out, segments_file=args.segments,
                          video_name=args.video_name)
    print(f"записано: {out_path}")
    print("назначение stretch их скриптом (после ручной загрузки данных, см. "
          "data/raw/videofruit/INSTRUCTIONS.txt):")
    print(f"  python external/VideoFruitCounting/assign_apples.py "
          f"--data_path {out_path.parent.parent} --sframe_file <SFRAME.csv>")
    return 0


def cmd_demo(args: argparse.Namespace, cfg: Config) -> int:
    from .data.synthetic import SyntheticParams, generate_synthetic
    from .pipeline import run_video

    out = Path(args.out)
    synth_dir = out / "synthetic"
    manifest_path = synth_dir / "gt" / "manifest.json"
    if args.force or not manifest_path.exists():
        params = SyntheticParams(n_trees=args.trees, frames=args.frames,
                                 seed=int(cfg.get("project.seed", 42)))
        generate_synthetic(str(synth_dir), params)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    summary = run_video(cfg, manifest["video"], str(out / "run"), gt_dir=str(synth_dir / "gt"))
    _print_summary(summary)
    print(f"\nотчёт: {summary.get('paths', {}).get('report')}")
    return 0


def cmd_journals(args: argparse.Namespace, cfg: Config) -> int:
    """Целевые журналы ВАК (перечень от 15.06.2026)."""
    rows: List[dict] = []
    csv_path = Path(__file__).resolve().parents[2] / "paper" / "journals_vak.csv"
    if csv_path.exists():
        rows = pd.read_csv(csv_path).to_dict("records")
    else:  # встроенный резервный список
        rows = [
            {"category": "К1", "name": "Искусственный интеллект и принятие решений",
             "issn": "2071-8594", "url": "https://raai.org/"},
            {"category": "К1", "name": "Программная инженерия", "issn": "2220-3397",
             "url": "https://www.sciencejournals.ru/journal/pi/"},
            {"category": "К1", "name": "Информатика и автоматизация", "issn": "2713-3192",
             "url": "https://www.iasa-ras.ru/"},
            {"category": "К1", "name": "Информационно-управляющие системы", "issn": "1684-8853",
             "url": "https://www.elbib.ru/journal.phtml?jrn=14"},
            {"category": "К2", "name": "Проблемы искусственного интеллекта",
             "issn": "2413-7383", "url": "https://ipiai.ru/"},
            {"category": "К2", "name": "Инженерный вестник Дона", "issn": "2073-8633",
             "url": "https://ivdon.ru/"},
            {"category": "К2", "name": "Вестник аграрной науки Донa", "issn": "2075-6704",
             "url": "https://donagrojour.ru/"},
            {"category": "К3", "name": "Агробиотехнологии и цифровое земледелие",
             "issn": "2782-490X", "url": "https://agrobiotech.ru/"},
        ]
    if args.category:
        rows = [r for r in rows if str(r.get("category", "")).upper() == args.category.upper()]
    print(f"{'катег.':6s} {'ISSN':12s} журнал")
    for r in rows:
        print(f"{str(r.get('category', '')):6s} {str(r.get('issn', '')):12s} {r.get('name', '')}")
    return 0


# ---------------------------------------------------------------------------
def _print_summary(summary: dict) -> None:
    print(f"кадров: {summary.get('frames')} | деревьев: {summary.get('n_trees')} | "
          f"уникальных яблок: {summary.get('total_apples')} "
          f"(без привязки: {summary.get('unassigned')})")
    cm = summary.get("count_metrics")
    if cm:
        print(f"счёт: расчёт {cm['pred_total']} / контроль {cm['gt_total']} "
              f"({cm['total_error_pct']:+.1f}%) | MAE {cm['mae']:.2f} | "
              f"в допуске {cm['within_tolerance']}/{cm['n_trees']}")
    dm = summary.get("detection_metrics")
    if dm:
        print(f"детекция: AP@{dm['iou_thr']:.2f}={dm['ap50']:.3f} "
              f"P={dm['precision']:.3f} R={dm['recall']:.3f} F1={dm['f1']:.3f}")
    tm = summary.get("tracking_metrics")
    if tm:
        print(f"трекинг: MOTA={tm['mota']:.3f} IDSW={tm['id_switches']}")
    y = summary.get("yield")
    if y:
        print(f"урожай: {y['total_kg']:.1f} кг [{y['ci_lo']:.1f} … {y['ci_hi']:.1f}]")


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging("DEBUG" if args.verbose else "INFO")
    try:
        cfg = load_config(args.config, args.set)
    except Exception as exc:
        logger.error("конфигурация: %s", exc)
        return 2
    try:
        return int(args.func(args, cfg))
    except KeyboardInterrupt:
        logger.warning("прервано пользователем")
        return 130
    except Exception as exc:
        logger.error("ошибка: %s", exc, exc_info=args.verbose)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
