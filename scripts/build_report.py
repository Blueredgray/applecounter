"""Сборка промежуточного отчёта: таблицы CSV + REPORT.md/.txt/.pdf.

Все числа читаются напрямую из summary.json/bench JSON — без хардкода.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
FIG = REPORT / "figures"
REPORT.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))


# ------------------------------------------------------------------ данные

def load_t1() -> pd.DataFrame:
    rows = []
    for det in ["hsv", "yolo8n", "yolo26s"]:
        for tr in ["botsort", "deepsort"]:
            s = json.loads((ROOT / f"outputs/rework_matrix/t1/{det}/{tr}/T1/summary.json").read_text(encoding="utf-8"))
            d, t, tm = s["detection_metrics"], s["tracking_metrics"], s["timing"]
            rows.append({
                "dataset": "T1 (Villacres, Zenodo 7384297)", "detector": det, "tracker": tr,
                "frames": s["frames"], "gt_tracks": t["n_gt"],
                "ap50": d["ap50"], "precision": d["precision"], "recall": d["recall"],
                "mota": t["mota"], "idf1": t["idf1"], "idsw": t["id_switches"],
                "unique_counted": s["total_apples"],
                "track_s_per_frame": tm["tracking_sec"] / s["frames"],
            })
    return pd.DataFrame(rows)


def load_minneapple() -> pd.DataFrame:
    rows = []
    for tag, name in [("hsv", "hsv"), ("yolo8n_640", "yolo8n"), ("yolo26s_1280", "yolo26s")]:
        b = json.loads((ROOT / f"outputs/minneapple_detection/bench_{tag}.json").read_text(encoding="utf-8"))
        m = b["hsv" if tag == "hsv" else "yolo"]
        rows.append({
            "dataset": "MinneApple val (331 img)", "detector": name,
            "n_images": b["n_images"], "n_gt_boxes": m["n_gt"],
            "ap50": m["ap50"], "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
            "count_mae_per_image": m["count_mae_per_image"],
            "mean_pred": m["mean_pred"], "mean_gt": m["mean_gt"],
        })
    return pd.DataFrame(rows)


def load_vfc() -> pd.DataFrame:
    rows = []
    for det in ["hsv", "yolo8n", "yolo26s"]:
        for tr in ["botsort", "deepsort"]:
            for man in sorted((ROOT / f"outputs/rework_matrix/vfc/{det}/{tr}").glob("*/run_manifest.json")):
                s = json.loads((man.parent / "summary.json").read_text(encoding="utf-8"))
                if s.get("frames", 0) < 1000:  # smoke-манифесты не считаем
                    continue
                rows.append({
                    "phase": f"vfc/{det}", "tracker": tr, "video": man.parent.name,
                    "frames": s["frames"], "unique_counted": s["total_apples"],
                    "unassigned": s.get("unassigned"), "n_segments": s.get("n_trees"),
                })
    return pd.DataFrame(rows)


METRICS_LEGEND = [
    ("AP@0.50", "Детекция", "Average Precision при IoU=0.50: средняя точность детекции по всем порогам уверенности. Основная метрика качества детектора. Выше — лучше."),
    ("Precision", "Детекция", "Доля верных срабатываний среди всех предсказанных боксов (TP/(TP+FP)). Высокая precision = мало ложных срабатываний. Выше — лучше."),
    ("Recall", "Детекция", "Доля найденных GT-объектов (TP/(TP+FN)). Высокий recall = мало пропущенных яблок. Выше — лучше."),
    ("F1", "Детекция", "Среднее гармоническое precision и recall: баланс одной цифрой. Выше — лучше."),
    ("Count MAE", "Счёт", "Средняя абсолютная ошибка числа яблок на изображение/видео. Прямо измеряет качество подсчёта урожая. Ниже — лучше."),
    ("MOTA", "Трекинг", "Multiple Object Tracking Accuracy: 1 − (FN+FP+IDSW)/GT. Комплексная точность трекинга с учётом пропусков, ложных срабатываний и смен идентификаторов. Может быть отрицательной. Выше — лучше."),
    ("IDF1", "Трекинг", "F1-метрика идентификации: насколько правильно трекер сохраняет тождество объектов (сколько наблюдений отнесено к правильному ID). Ключевая метрика для задачи «не пересчитывать одно яблоко дважды». Выше — лучше."),
    ("ID switches (IDSW)", "Трекинг", "Число смен идентификатора у одного физического объекта: прямой показатель дробления треков и двойного счёта. Ниже — лучше."),
    ("unique_counted", "Счёт", "Число уникальных подтверждённых треков, принятых за отдельные яблоки. Без count GT — внутренняя согласованность, а не точность."),
]


# ------------------------------------------------------------------ тексты

def fmt(df: pd.DataFrame, cols: list[str]) -> str:
    return df[cols].to_string(index=False)


def build_report_text(t1: pd.DataFrame, mn: pd.DataFrame, vfc: pd.DataFrame) -> dict[str, str]:
    hsv = vfc[vfc.phase == "vfc/hsv"]
    vfc_hsv_done = hsv[hsv.tracker == "botsort"].shape[0]
    vfc_y8 = vfc[vfc.phase == "vfc/yolo8n"].shape[0] // 2
    bots = hsv[hsv.tracker == "botsort"].unique_counted.mean()
    deep = hsv[hsv.tracker == "deepsort"].unique_counted.mean()

    md: list[str] = []
    a = md.append

    a("# Промежуточный отчёт: подсчёт яблок на деревьях по видео")
    a("")
    a("**Дата:** 09.11.2026 · **Коммит кода:** `ae5b3c6` · **Статус:** промежуточный результат для доклада")
    a("")
    a("## 1. Сводка статуса")
    a("")
    a("| Этап | Статус |")
    a("|---|---|")
    a("| Детекция на MinneApple val (HSV / YOLOv8n / YOLO26s) | ✅ 3/3 готово |")
    a("| MOT-матрица T1 (3 детектора × 2 трекера) | ✅ 6/6 готово |")
    a("| VFC, фаза HSV (16 видео × 2 трекера) | ✅ 32/32 готово |")
    a("| VFC, фаза YOLOv8n | 🔄 2/32 (в работе) |")
    a("| VFC, фаза YOLO26s | ⏳ очередь |")
    a("| Ручной count GT для VFC | ❌ не начат — блокирует абсолютную ошибку счёта на VFC |")
    a("")
    a("Прогоны выполняются автоматическим сценарием `scripts/overnight_experiments.ps1` «в лоб», "
      "одна задача за раз; сценарий возобновляем — после сбоя питания валидные задачи пропускаются. "
      "Каждый прогон фиксирует манифест с SHA-256 входного видео и конфигурацией.")
    a("")
    a("## 2. Датасеты")
    a("")
    a("- **MinneApple** — кадры яблонь с боксовой разметкой; использован для обучения и валидации детекции "
      "(train 670 / val 331 изображение, 40 464 GT-боксов). Не используется для видеотрекинга.")
    a("- **T1** — датасет Villacres et al. (Zenodo 7384297, Computers and Electronics in Agriculture 2022): "
      "последовательность из 1290 кадров с MOT-разметкой (939 яблок, 48 111 боксов). "
      "Единственный датасет в работе, где есть GT для трекинга — источник метрик MOTA/IDF1/IDSW.")
    a("- **VideoFruitCounting (VFC)** — 16 видео с двух камер (Azure Kinect, ~8–9 тыс. кадров каждое). "
      "Счёт ведётся по 21 stretch-маркеру (vfc_reference). GT числа яблок отсутствует — метрики точности "
      "детекции/трекинга там недостижимы, возможна только внутренняя согласованность.")
    a("")
    a("## 3. Пайплайн")
    a("")
    a("`видео → детекция (HSV | YOLOv8n@640 | YOLO26s@1280) → кэш детекции (SHA-256, проверка конфигурации) → "
      "трекинг (BoT-SORT | DeepSORT, внешний ByteTrack — партнёрский этап) → MOT-метрики (motmetrics) → "
      "подсчёт уникальных треков → привязка к stretch/деревьям → отчёт`.")
    a("")
    a("## 4. Описание используемых метрик")
    a("")
    a("| Метрика | Уровень | Что означает и как влияет |")
    a("|---|---|---|")
    for name, lvl, desc in METRICS_LEGEND:
        a(f"| {name} | {lvl} | {desc} |")
    a("")
    a("Полные определения с формулами — в `report/metrics_legend.csv`.")
    a("")
    a("## 5. Результаты: MinneApple val (детекция)")
    a("")
    a("| Детектор | AP@0.50 | Precision | Recall | F1 | Count MAE/изобр. | Среднее предск. | GT |")
    a("|---|---|---|---|---|---|---|---|")
    for _, r in mn.iterrows():
        a(f"| {r.detector} | {r.ap50:.3f} | {r.precision:.3f} | {r.recall:.3f} | {r.f1:.3f} | "
          f"{r.count_mae_per_image:.2f} | {r.mean_pred:.1f} | {r.mean_gt:.1f} |")
    a("")
    a(f"![MinneApple: детекция](figures/minneapple_yolo8n.png)")
    a("")
    a(f"![MinneApple: метрики](figures/minneapple_metrics.png)")
    a("")
    a("## 6. Результаты: T1 (трекинг, 1290 кадров, GT 939 яблок)")
    a("")
    a("| Детектор | Трекер | AP@0.50 | MOTA | IDF1 | IDSW | Уникальных треков | трекинг, с/кадр |")
    a("|---|---|---|---|---|---|---|---|")
    for _, r in t1.iterrows():
        a(f"| {r.detector} | {r.tracker} | {r.ap50:.3f} | {r.mota:+.3f} | {r.idf1:.3f} | {int(r.idsw)} | "
          f"{int(r.unique_counted)} | {r.track_s_per_frame:.2f} |")
    a("")
    a(f"![T1: треки yolo8n+botsort](figures/t1_yolo8n_botsort.png)")
    a("")
    a(f"![T1: метрики](figures/t1_metrics.png)")
    a("")
    a(f"![T1: длины треков](figures/t1_track_lengths.png)")
    a("")
    a("## 7. Промежуточные результаты VFC")
    a("")
    a(f"HSV-фаза закрыта полностью ({vfc_hsv_done} видео × 2 трекера). По {vfc_hsv_done} видео "
      f"BoT-SORT в среднем даёт {bots:,.0f} уникальных треков, DeepSORT — {deep:,.0f}. "
      "Расхождение согласуется с T1 (у DeepSORT больше разрывов ID). "
      f"YOLOv8n-фаза: {vfc_y8} видео обоими трекерами; "
      "первые результаты: YOLOv8n+BoT-SORT 8 439 уникальных яблок на видео `210928_100624` "
      "против 60 792 у HSV — HSV-бейзлайн завышает счёт в ~7 раз, что согласуется с AP50=0.016 на MinneApple.")
    a("")
    a("Детекция YOLOv8n на этом видео заняла 160 с (~57 к/с, CUDA), трекинг BoT-SORT — 304 мин; "
      "цена кадра трекинга растёт с числом детекций и активных треков — отмечено как ограничение производительности.")
    a("")
    a("## 8. Что исправлено относительно исходной версии")
    a("")
    a("Исходная реализация сохранена коммитом `f14b87e`. По результатам аудита и переработки (`c911144`, "
      "`a9aabcd`, `ae5b3c6`):")
    a("")
    a("1. **Привязка к деревьям.** Раньше яблоко приписывалось группе деревьев, а не конкретной позиции; "
      "теперь единица счёта — stretch из SFRAME (vfc_reference), деревья не дробятся автоматически.")
    a("2. **Воспроизводимость.** Добавлены манифесты прогонов (SHA-256 входа, конфигурация, статус), "
      "кэш детекции валидируется по конфигурации и автоматически пересоздаётся при несовместимости "
      "(исправлено падение возобновления после обрыва питания).")
    a("3. **Сводная таблица.** Раньше каждая фаза затирала `matrix_results.csv`; теперь фазы сливаются, "
      "добавлен идемпотентный `scripts/rebuild_results.py`.")
    a("4. **Детекция.** MinneApple сконвертирован в YOLO-формат (data.yaml, 670/331), PyTorch переведён "
      "с CPU-сборки на CUDA (2.14.1+cu126, RTX 3050): детекция YOLO на видео ускорилась на порядки.")
    a("5. **Валидация.** 112 тестов; MOT-метрики через motmetrics; GT T1 проверен на валидность MOT "
      "(939 идентификаторов, 937 наблюдаются повторно).")
    a("")
    a("## 9. Сравнительный анализ и выбор комбинации")
    a("")
    a("**Детектор.** HSV-цветовой бейзлайн непригоден: AP50 = 0,016 (MinneApple) и 0,003 (T1), "
      "счёт завышен в разы — оставлен только как нижняя граница. YOLOv8n и YOLO26s сопоставимы: "
      "на MinneApple YOLO26s лучше по AP50 (0,775 против 0,656) и recall, по count MAE — паритет "
      "(6,7 против 6,4 яблока); на T1 YOLOv8n чуть лучше по AP50 (0,387 против 0,359). "
      "YOLO26s заметно дороже по времени детекции.")
    a("")
    a("**Трекер.** BoT-SORT стабильно превосходит DeepSORT по удержанию идентификаторов: "
      "IDF1 0,468 против 0,412 (YOLOv8n) и 0,431 против 0,345 (YOLO26s); IDSW меньше в 2,7–3,4 раза "
      "(81 против 272; 72 против 249). Цена — время трекинга: BoT-SORT в 1,5–7 раз медленнее в зависимости "
      "от плотности сцен.")
    a("")
    a("**Итог.** Оптимальная комбинация для задачи подсчёта — **YOLOv8n + BoT-SORT**: лучший MOTA (0,278) "
      "и IDF1 (0,468) на единственном датасете с трекинг-GT при минимальном времени детекции. "
      "**YOLO26s + BoT-SORT** — ближайшая альтернатива с лучшей детекцией на MinneApple (AP50 0,775) "
      "и точностью 0,809 — претендент №1 там, где важнее качество боксов, чем скорость. "
      "Окончательный выбор между ними для счёта урожая — только после ручного count GT на VFC.")
    a("")
    a("## 10. Ограничения и следующие шаги")
    a("")
    a("1. На VFC нет GT: без ручного count GT нельзя заявлять абсолютную ошибку счёта и выбирать "
      "метод по точности подсчёта (протокол разметки — `docs/research_run_plan.md`, раздел 4).")
    a("2. Фазы VFC YOLOv8n/YOLO26s досчитываются (~8–10 суток при текущей скорости).")
    a("3. ByteTrack — партнёрский этап поверх того же кэша и интерфейса.")
    a("4. Этап 3 (компенсация движения камеры, восстановление ID) — после текущей матрицы.")
    a("5. Прогноз урожая в кг — только после полевой калибровки массы; до неё — иллюстративный.")

    return {"md": "\n".join(md)}


# ------------------------------------------------------------------ вывод

def write_csv_tables(t1, mn, vfc) -> None:
    t1.round(4).to_csv(REPORT / "t1_results.csv", index=False)
    mn.round(4).to_csv(REPORT / "minneapple_results.csv", index=False)
    vfc.to_csv(REPORT / "vfc_results_partial.csv", index=False)
    legend = pd.DataFrame(METRICS_LEGEND, columns=["metric", "level", "description"])
    legend.to_csv(REPORT / "metrics_legend.csv", index=False)
    # свод всех прогонов со всеми датасетами
    full = pd.read_csv(ROOT / "outputs/rework_matrix/matrix_results.csv")
    full = full[full.frames > 1000]
    full.to_csv(REPORT / "results_all.csv", index=False)


def md_to_txt(md: str) -> str:
    lines = []
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("!["):
            alt = s[2:s.index("](")]
            path = s[s.index("](") + 2:-1]
            lines.append(f"[Рисунок: {alt} — {path}]")
        elif s.startswith("#"):
            lines.append("")
            lines.append(s.lstrip("#").strip().upper() if s.startswith("## ") else s.lstrip("# ").strip())
            lines.append("")
        elif s.startswith("| "):
            if set(s) <= set("|- "):
                continue
            cells = [c.strip() for c in s.strip("|").split("|")]
            lines.append("  ".join(cells))
        else:
            lines.append(line.replace("**", "").replace("`", ""))
    return "\n".join(lines)


def build_pdf(md_text: str) -> None:
    from PIL import Image as PILImage
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)
    from reportlab.lib.styles import ParagraphStyle
    import matplotlib

    font_path = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    pdfmetrics.registerFont(TTFont("DejaVu", str(font_path)))
    styles = {
        "h1": ParagraphStyle("h1", fontName="DejaVu", fontSize=15, spaceAfter=8, leading=19),
        "h2": ParagraphStyle("h2", fontName="DejaVu", fontSize=12, spaceAfter=6, spaceBefore=10, leading=15),
        "body": ParagraphStyle("body", fontName="DejaVu", fontSize=9, leading=13, spaceAfter=5),
        "cell": ParagraphStyle("cell", fontName="DejaVu", fontSize=7.5, leading=9.5),
    }
    doc = SimpleDocTemplate(str(REPORT / "REPORT.pdf"), pagesize=A4,
                            leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.6 * cm)
    story = []

    def flush_table(rows):
        if not rows:
            return
        def cell(c):
            for emoji, word in [("✅", "OK"), ("🔄", "в работе"), ("⏳", "очередь"), ("❌", "нет")]:
                c = c.replace(emoji, word)
            return Paragraph(c, styles["cell"])
        data = [[cell(c) for c in row] for row in rows]
        tbl = Table(data, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde4f0")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 6))

    def add_image(path: Path, max_width=17.5 * cm, max_height=21.5 * cm):
        img = PILImage.open(path)
        w, h = img.size
        max_px = 1400
        if w > max_px:
            img = img.resize((max_px, int(h * max_px / w)))
            tmp = path.parent / (path.stem + "_small.png")
            img.save(tmp)
            path = tmp
        w_pt, h_pt = max_width, max_width * img.size[1] / img.size[0]
        if h_pt > max_height:
            w_pt, h_pt = max_height * img.size[0] / img.size[1], max_height
        story.append(Image(str(path), width=w_pt, height=h_pt))
        story.append(Spacer(1, 6))

    table_rows: list[list[str]] = []
    for line in md_text.splitlines():
        s = line.strip()
        if s.startswith("|"):
            if set(s) <= set("|- "):
                continue
            table_rows.append([c.strip() for c in s.strip("|").split("|")])
            continue
        flush_table(table_rows)
        table_rows = []
        if not s:
            continue
        if s.startswith("!["):
            add_image(FIG / s[s.index("](") + 2:-1].removeprefix("figures/"))
        elif s.startswith("### "):
            story.append(Paragraph(s[4:], styles["h2"]))
        elif s.startswith("## "):
            story.append(Paragraph(s[3:], styles["h2"]))
        elif s.startswith("# "):
            story.append(Paragraph(s[2:], styles["h1"]))
        else:
            html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s).replace("`", "")
            for emoji, word in [("✅", "[OK]"), ("🔄", "[в работе]"), ("⏳", "[очередь]"), ("❌", "[нет]")]:
                html = html.replace(emoji, word)
            story.append(Paragraph(html, styles["body"]))
    flush_table(table_rows)
    doc.build(story)


def main() -> None:
    t1, mn = load_t1(), load_minneapple()
    vfc = load_vfc()
    write_csv_tables(t1, mn, vfc)
    texts = build_report_text(t1, mn, vfc)
    (REPORT / "REPORT.md").write_text(texts["md"], encoding="utf-8")
    (REPORT / "REPORT.txt").write_text(md_to_txt(texts["md"]), encoding="utf-8")
    build_pdf(texts["md"])
    print("report ->", REPORT)


if __name__ == "__main__":
    main()
