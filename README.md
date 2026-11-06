# AppleCounter — подсчёт яблок и прогноз урожайности по видео пролёта БПЛА

Полный конвейер компьютерного зрения: детекция яблок → отслеживание без
повторов → привязка к деревьям ряда → прогноз массы урожая с доверительным
интервалом → метрики и отчёт (Markdown + LaTeX + PDF).

## Возможности

- **Детекция**: цветовой HSV-детектор (без зависимостей) и опционально
  нейросетевой YOLO (`ultralytics`), обучаемый на MinneApple / ACFR.
- **Стабилизация**: ORB-смещения накапливаются, наблюдения переводятся в
  мировые координаты ряда — движение камеры не ломает трекинг.
- **Трекинг**: IoU-этап + re-ID по внешнему виду с пространственным гейтом
  (иначе плоды соседних деревьев «перетекают» друг в друга). Подключаемы
  три трекера с единым интерфейсом (`--set tracker.name=...`):
  `ours` (собственный), `botsort` (BoT-SORT: КФ + двухэтапная ассоциация
  high/low-conf + re-ID), `deepsort` (DeepSORT: КФ + каскадная ассоциация
  движение→ReID-галерея).
- **Подсчёт без повторов**: учитывается только подтверждённый трек
  (CONFIRMED, ≥ `min_obs` наблюдений), фрагменты одного плода сливаются
  (`counting.merge_dist`), события двойного счёта равны нулю.
- **Деревья**: ряд сегментируется по проекции мировых координат на ось PCA +
  плотность/просветы (`trees.method: axis|segments|boundaries|fixed`).
- **Урожай**: физическая модель `M = N·m/κ` с 95 % ДИ и линейная регрессия
  `M = a + bN` по историческим данным (`yieldpred/model.py`).
- **Метрики**: AP@IoU, PR-кривая, MOTA, подмены ID, MAE/RMSE/MAPE/`R²` счёта
  по деревьям, число деревьев в допуске ±3.
- **Отчёт**: `report.md`, графики, мозаика прогона, `annotated.mp4`,
  LaTeX-таблицы `results_table.tex` / `results_yield.tex` для статьи.

## Установка

```powershell
python -m pip install -e .          # пакет + CLI applecounter
python -m pip install -e ".[yolo]"  # опционально: ultralytics + torch (YOLO)
python -m pip install pytest        # тесты
```

Python ≥ 3.9; зависимости: numpy, opencv-python, scipy, pandas, PyYAML,
matplotlib.

## Быстрый старт

```powershell
$env:PYTHONIOENCODING = "utf-8"     # кириллица в логах Windows
applecounter demo --out outputs\demo
```

`demo` генерирует синтетический сад с эталоном, прогоняет весь конвейер и
печатает метрики, например:

```
счёт: расчёт 155 / контроль 154 (+0.6%) | MAE 0.17 | в допуске 6/6
детекция: AP@0.50=0.957 P=0.916 R=0.967 F1=0.941
трекинг: MOTA=0.888 IDSW=5
урожай: 27.9 кг [22.2 … 33.6]
```

## Команды

| Команда    | Назначение |
|------------|------------|
| `synth`    | сгенерировать синтетическое видео с ground truth |
| `download` | скачать датасеты (MinneApple, ACFR; инструкция для VideoFruitCounting) |
| `prepare`  | конвертировать маски в формат YOLO (+ `data.yaml`) |
| `train`    | дообучить YOLO на подготовленном датасете |
| `run`      | полный проход по видео: детекция → трекинг → счёт → отчёт |
| `evaluate` | оценить готовый прогон по ground truth |
| `bench`    | бенчмарк детекторов (hsv/yolo) на val-сплите датасета |
| `yield`    | прогноз урожая по `counts.csv` |
| `report`   | пересобрать отчёт и графики из сохранённого прогона |
| `vfc-export` | экспорт прогона в формат VideoFruitCounting (`all_predictions.csv`) |
| `demo`     | демо end-to-end на синтетике |
| `journals` | целевые журналы ВАК (`--category К1` и т.п.) |

Общие опции: `--config CONFIG` (по умолчанию `configs/default.yaml`),
`--set ключ=значение` (например `--set tracker.reid_max_dist=60`),
`-v` — подробный лог.

Пример реального прогона:

```powershell
python -m applecounter run --video flight.mp4 --out outputs\flight --set detector=hsv
python -m applecounter evaluate --run outputs\flight --gt gt/
python -m applecounter yield --run outputs\flight --mean-mass 180
```

## Конфигурация

Все параметры — в `configs/default.yaml`: детектор, стабилизация, трекинг
(`tracker.name`: ours|botsort|deepsort, `tracker.min_obs`,
`tracker.reid_max_dist`), сегментация деревьев (`trees.method`,
`trees.gap_ratio`), подсчёт (`counting.merge_dist`), урожай
(`yield.mean_mass_g`, `yield.visibility_factor`), метрики и вывод.

## Метод (кратко)

1. Кадры → ключевые точки ORB → накопленное смещение камеры → мировые
   координаты ряда.
2. Детекция рамок (HSV или YOLO) в мировых координатах.
3. Трекинг: IoU вблизи, затем re-ID по цветовой гистограмме — только в
   радиусе `reid_max_dist` мировых пикселей.
4. Ряд сегментируется на деревья (PCA + плотность + просветы).
5. Каждый подтверждённый трек считается один раз; осколки одного плода
   сливаются по близости (`merge_dist`).
6. Счёт по деревьям → прогноз урожая (точка + 95 % ДИ).

## Результаты (контрольный прогон, синтетика)

427 кадров, 6 деревьев, эталон 154 плода; 13 с на ЦП (~33 к/с):

| Метрика | Значение |
|---------|----------|
| Итоговый счёт | 155 vs 154 (+0,65 %) |
| Деревьев в допуске ±3 | 6/6 (5 точно) |
| MAE / RMSE / MAPE | 0,17 / 0,41 / 0,6 % |
| AP@0.50 (P / R / F1) | 0,957 (0,916 / 0,967 / 0,941) |
| MOTA / IDSW | 0,888 / 5 |
| Урожай | 27,9 кг [22,2 … 33,6] |

## Результаты (реальные данные, ACFR MultiFruit 2016)

- Датасет скачан и сконвертирован: **1120 изображений → YOLO train 896 / val 224**
  (5765 боксов из `annotations/*.csv`; сплиты из штатных `sets/`).
- Дообучение YOLOv8n (5 эпох, RTX 3050): val **P=0,89, R=0,84, mAP50=0,933,
  mAP50-95=0,659** → `weights/runs/acfr/weights/best.pt`.
- Прогон на видео из кадров ACFR (250 кадров, `outputs/acfr_yolo`):
  **57 яблок, 10,3 кг [6,1 … 14,4]** (гейт re-ID 250 px, стабилизация off —
  кадры исходно сняты с большим интервалом). На тех же кадрах HSV-детектор
  даёт 2 яблока — обоснование нейросетевого детектора (`outputs/acfr_hsv`).

## Результаты (реальные данные, MinneApple, COCO-вариант)

Датасет MinneApple (HuggingFace, COCO-JSON) добавлен в `data/raw/minneapple/`
и сконвертирован в YOLO одной командой:

```powershell
applecounter prepare --dataset minneapple --src data/raw/minneapple --out data/processed/minneapple
```

- **Конвертация**: 670 train / 331 val изображения (720×1280), **40 464 бокса**
  из COCO-JSON, 0 пустых меток; сплиты датасета train/test → train/val.
- **Дообучение YOLOv8n** (50 эпох, RTX 3050, ~10 мин): val **P=0,778,
  R=0,654, mAP50=0,742, mAP50-95=0,400** → `weights/runs/minneapple50/weights/best.pt`
  (5 эпох: mAP50=0,654; 20 эпох: 0,732 — рост к 50 эпохам замедляется).
- **Бенчмарк детекторов на val** (`applecounter bench`, наши метрики,
  `outputs/minneapple/bench.json` и `outputs/minneapple/bench_yolo26s1280.json`):

| Детектор | AP@0.50 | P | R | F1 | count-MAE / кадр |
|----------|--------:|--:|--:|--:|-----------------:|
| HSV      | 0,016 | 0,130 | 0,120 | 0,125 | 13,24 |
| YOLOv8n @640 | 0,656 | 0,759 | 0,698 | 0,727 |  6,41 |
| **YOLO26s @1280** | **0,775** | 0,740 | **0,817** | **0,777** | **6,66** |

  (датасет сильно засорён: ~42 яблока/кадр, плотные заросли;
  HSV-пороги не покрывают зрелые плоды MinneApple.)

- **Апгрейд детектора: YOLO26s @1280** (`weights/runs/minneapple26s1280/`):
  40 эпох, RTX 3050, `--set train.workers=2` (OOM в DataLoader при 4).
  Ultralytics val: **P=0,81, R=0,754, mAP50=0,844, mAP50-95=0,521** против
  v8n@640 (0,778 / 0,654 / 0,742 / 0,400): **+0,10 mAP50, +0,12 mAP50-95**.

- **Видео-прогон с эталоном** (`scripts/minneapple_video.py` собирает mp4 из
  последовательности `20150919_174730`: 100 кадров 720×1280, 3118 GT-боксов,
  31,2 яблока/кадр; per_frame GT — из COCO-аннотаций):

| Детектор | AP@0.50 | P | R | F1 | MOTA | IDSW | счёт | урожай |
|----------|--------:|--:|--:|--:|-----:|-----:|-----:|--------|
| YOLO (minneapple50) | 0,755 | 0,889 | 0,772 | 0,826 | 0,215 | 0 | 171 | 30,8 кг [22,5 … 39,0] |
| HSV | 0,004 | 0,108 | 0,033 | 0,051 | 0,000 | 0 | 19 | 3,4 кг |

  MOTA низок из-за пропусков детекции в плотных кадрах (R=0,77), но IDSW=0 —
  трекер не «путает» плоды; precision 0,89 при огромной плотности. Каталоги:
  `outputs/minneapple_video/` (YOLO), `outputs/minneapple_video_hsv/` (HSV).

### Сравнение трекеров: ours vs BoT-SORT vs DeepSORT

BoT-SORT (Zhang et al., 2022) и DeepSORT (Wojke et al., 2017) реализованы
в проекте (`track/kalman.py`, `track/botsort.py`, `track/deepsort.py`):
общий Калмановский фильтр бокса (u, v, w, h + скорости) + разные схемы
ассоциации. ByteTrack — отдельная реализация другого соавтора, единый
интерфейс трекера позволяет подключить его тем же ключом
`--set tracker.name=bytetrack` (контракт интерфейса и пошаговый гайд —
`docs/adding_a_tracker.md`).

Синтетика (детектор HSV, эталон 154 плода):

| Трекер | Счёт | MAE | Деревья в допуске | MOTA | IDSW | Создано треков |
|--------|-----:|----:|------------------:|-----:|-----:|---------------:|
| ours | 155 | 0,17 | 6/6 | 0,888 | 13 | 180 |
| BoT-SORT | 153 | 0,17 | 6/6 | 0,887 | **8** | **160** |
| DeepSORT | 153 | 0,17 | 6/6 | 0,887 | **8** | **160** |

Реальное видео MinneApple, детектор YOLOv8n (AP 0,755), одинаковые детекции:

| Трекер | MOTA | IDSW | Счёт | Урожай |
|--------|-----:|-----:|-----:|--------|
| ours | 0,215 | 0 | 171 | 30,8 кг |
| BoT-SORT | **0,516** | 0 | 226 | 40,7 кг |
| DeepSORT | 0,454 | 0 | 204 | 36,7 кг |

То же видео, детектор **YOLO26s@1280** (AP 0,874), одинаковые детекции:

| Трекер | MOTA | IDSW | Счёт | Урожай | Создано треков |
|--------|-----:|-----:|-----:|--------|---------------:|
| ours | 0,301 | 0 | 262 | 47,2 кг | 1632 |
| BoT-SORT | 0,620 | 0 | 282 | 50,8 кг | 476 |
| DeepSORT | **0,629** | 0 | 246 | 44,3 кг | **445** |

Апгрейд детектора поднял MOTA всех трекеров на 0,1–0,3 и счёт ближе к GT;
на сильном детекторе BoT-SORT и DeepSORT сравнялись (0,62 против 0,63).
На плотных реальных кадрах BoT-SORT/DeepSORT почти вдвое снижают фрагментацию
треков (1632 → ~450 создано треков) и поднимают MOTA за счёт КФ-предсказания
и двухэтапной ассоциации; каталоги: `outputs/synth_track_{ours,botsort,deepsort}`,
`outputs/minneapple_video_{botsort,deepsort}`,
`outputs/minneapple_video_26s{,_ours,_deepsort}`.

### Этап 1 — VideoFruitCounting: метод stretches (Felip-Pomés et al., 2026)

Конвейер VideoFruitCounting (https://github.com/imatge-upc/VideoFruitCounting):
YOLOv5 → ByteTrack → `all_predictions.csv` → `assign_apples.py` распределяет
яблока по **stretch'ам** — отрезкам ряда, размеченным вертикальными
лентами-маркерами; SFRAME-файл даёт кадры, в которые маркер оказывается по
центру кадра. Воспроизведение метода в нашем проекте:

- **границы по кадрам уже поддерживаются**: `trees.method: segments` +
  `trees.segments_file` принимает их SFRAME-формат `<video>,f0,f1,...,fN`
  (функция `load_segments`), привязка трека к stretch'у — по медиане кадров
  наблюдений, что для монотонного пролёта эквивалентно их majority vote;
- **экспорт в их формат**: `applecounter run` сохраняет `tracks_frame.csv`
  (экран-боксы подтверждённых треков по кадрам), а команда

  ```powershell
  applecounter vfc-export --run outputs\flight --segments SFRAME.csv
  ```

  собирает `outputs/flight/vfc/<видео>/all_predictions.csv` — структуру,
  которую напрямую принимает их скрипт:

  ```powershell
  python external/VideoFruitCounting/assign_apples.py --data_path outputs\flight\vfc --sframe_file SFRAME.csv
  ```

- **данные скачаны вручную** (gofile требует входа) в
  `data/raw/videofruit/Video Fruit Counting/`: 16 MP4 (8 Azure Kinect + 8 ZED 2,
  1920×1080 @30 fps, ~9000 кадров, кадры сняты вертикально и лежат «на боку») —
  поворот зависит от установки камеры: **Kinect `--set video.rotate90=-1`
  (против часовой), ZED `--set video.rotate90=1` (по часовой)**; `stretch_sframe.csv`
  (SFRAME: 124 видео × 22 границы = 21 stretch) и их веса YOLOv5 `last.pt`
  (лицензия AGPL — в MIT-код проекта не копируются); сам код
  клонирован в `external/VideoFruitCounting/`. Прогон на реальных данных:

  ```powershell
  python -m applecounter run --video "data/raw/videofruit/Video Fruit Counting/Videos/Azure Kinect/210928_100624_k_r1_e_015_225_162.mp4" `
    --out outputs/vfc_k_r1_e --set video.rotate90=-1 `
    --set detector.name=yolo --set detector.yolo.weights=weights/runs/minneapple26s1280/weights/best.pt `
    --set detector.yolo.imgsz=1280 --set trees.method=segments `
    --set trees.segments_file="data/raw/videofruit/Video Fruit Counting/stretch_sframe.csv" `
    --set tracker.name=botsort
  ```

  Результат первого видео (9040 кадров за 37 мин, RTX 3050): 21 stretch,
  **9815 уникальных яблок** (445 без привязки — подъезд к ряду до первого
  маркера), 0 событий двойного счёта, урожай 1686,6 кг [1564,0 … 1809,2];
  счёт по stretch'ам — 201…622 (медиана ≈ 457). Отчёт: `outputs/vfc_k_r1_e/report.md`.

  **Все 16 видео прогнаны** (`scripts/run_vfc_all.ps1`, ~9,3 ч на RTX 3050):
  131 994 кадра, **120 269 уникальных яблок**, 0 событий двойного счёта,
  20 865 кг урожая (Kinect 84 817 яблок / 14 716 кг; ZED 35 452 / 6 149 кг).
  Сводная таблица: `outputs/vfc_summary.csv`. Готовые прогоны пропускаются,
  `-Only <подстрока>` — фильтр; поворот выбирается по имени видео
  (Kinect `rotate90=-1`, ZED `rotate90=1`) и фиксируется в `rot.txt`.

- **сквозная проверка моста**: их `assign_apples.py` принимает наш экспорт
  напрямую (`--data_path outputs\vfc_k_r1_e\vfc`) и на выходе даёт счёт по
  stretch'ам, коррелированный с нашим на **0,92** (их 10 423 против наших
  9 370 назначенных; расхождение — другая логика: они назначают stretch по
  кадру границы + половине кадра, мы — по границе трека). Результат:
  `outputs/vfc_k_r1_e/results_stretch/`.

- **ByteTrack** — их трекер; в разделе сравнения трекеров он подключается тем
  же ключом `--set tracker.name=bytetrack` (реализация другого соавтора).

## Датасеты

| Датасет | Команда | Статус в репозитории |
|---------|---------|----------------------|
| ACFR MultiFruit 2016 (2,2 ГБ) | `applecounter download --dataset acfr --out data/raw` | ✅ скачан, распакован, сконвертирован в YOLO (896/224) |
| MinneApple (COCO, HuggingFace) | `applecounter download --dataset minneapple --out data/raw` | ✅ скачан вручную (HF `data/raw/minneapple`), сконвертирован в YOLO (670/331, 40 464 бокса) |
| VideoFruitCounting | `applecounter download --dataset videofruit --out data/raw` | ✅ скачан вручную (`data/raw/videofruit/Video Fruit Counting/`: 16 MP4, SFRAME на 124 видео, веса YOLOv5) |

```powershell
applecounter download --list
applecounter prepare --dataset acfr --src data/raw/acfr --out data/processed/acfr
applecounter prepare --dataset minneapple --src data/raw/minneapple --out data/processed/minneapple
```

- MinneApple: https://conservancy.umn.edu/handle/11299/206575 (при 403 —
  скачать вручную и указать `--archive`).
- ACFR: https://data.acfr.usyd.edu.au/ag/treecrops/2016-multifruit/
- VideoFruitCounting: ссылки в `INSTRUCTIONS.txt` (видео, веса YOLOv5, SFRAME).

## Тесты

```powershell
python -m pytest
```

97 тестов: геометрия, конфигурация, детекторы, трекер (ours/BoT-SORT/DeepSORT),
счётчик, сегментация деревьев, стабилизация, синтетика, конвертеры (в т.ч.
реальные имена ACFR и COCO-MinneApple), экспорт в формат VideoFruitCounting,
бенчмарк CLI, урожай, пайплайн (≈ 25 с).

## Структура

```
configs/            default.yaml, train_yolo.yaml
src/applecounter/
  cli.py            командная строка (synth/download/prepare/train/run/...)
  pipeline.py       run_video, evaluate_run, артефакты и отчёт
  detect/           base, color_hsv, yolo
  track/            appearance, tracker (ours), kalman (КФ), botsort, deepsort
  trees/            stabilize (мировые координаты), segment (карта деревьев)
  counting/         counter (учёт, merge осколков)
  yieldpred/        model (физическая модель, регрессия, ДИ)
  eval/             metrics (AP/MOTA/счёт)
  viz/              draw, report (графики, mosaic, report.md, LaTeX)
  data/             synthetic (генератор GT), download, convert (YOLO)
tests/              83 pytest-теста
paper/              статья на русском (LaTeX), references.bib, журналы ВАК
scripts/            вспомогательные скрипты (minneapple_video.py — mp4+GT из COCO)
data/               датасеты (README в каждом подкаталоге)
  raw/acfr/         ✅ ACFR MultiFruit скачан (2,2 ГБ, в .gitignore)
  raw/minneapple/   ✅ MinneApple COCO-вариант (HuggingFace, в .gitignore)
  processed/acfr/   ✅ YOLO 896/224 + acfr_frames.mp4 (в .gitignore)
  processed/minneapple/ ✅ YOLO 670/331, 40 464 бокса (в .gitignore)
  processed/minneapple_video/ ✅ mp4 100 кадров + per_frame GT из COCO (в .gitignore)
  synthetic/        сгенерированный сад + эталон GT (в .gitignore)
  yield_history.csv пример полевого контроля для регрессии урожая
outputs/            результаты прогонов (README со структурой артефактов)
  demo/             applecounter demo: synthetic/ + run/ (полный пайплайн)
  synthetic_orchard/ run + evaluate + yield + report на data/synthetic
  acfr_yolo/        реальное видео ACFR + YOLO (57 яблок)
  acfr_hsv/         то же видео + HSV (для сравнения)
  minneapple/       bench.json: HSV vs YOLO на val MinneApple
  minneapple_video/ видео 100 кадров + YOLO (AP 0.755, счёт 171)
  minneapple_video_hsv/ то же видео + HSV (для сравнения)
weights/            yolov8n.pt + runs/acfr (mAP50 0.933) + runs/minneapple50 (mAP50 0.742)
```

## Статья

`paper/article.tex` — PDF собирается дважды:

```powershell
cd paper
pdflatex -interaction=nonstopmode article.tex   # ×2
```

Таблицы `results_table.tex` / `results_yield.tex` и графики
`paper/figures/` копируются из `outputs/<прогон>/` (`report`/`run`).
Список целевых журналов ВАК: `paper/journals_vak.csv`
(команда `applecounter journals`), рекомендации по подаче — в
`paper/README.md`.

## Лицензия

MIT.
