# AppleCounter

Воспроизводимый pipeline для детекции и подсчёта яблок на видео: HSV/YOLO →
BoT-SORT/DeepSORT/ByteTrack → уникальные track ID → участки ряда → оценка массы
при наличии полевой калибровки.

> Старые результаты в `outputs/` сохранены как исторические артефакты, но не
> используются новой матрицей. Прежние значения для MinneApple video и
> автоматически разбитых «деревьев» VFC нельзя переносить в статью: там не было
> корректных постоянных GT ID и размеченных границ отдельных деревьев.

## Что реализовано

- **Детекция:** HSV baseline и Ultralytics YOLO. Веса текущих экспериментов:
  YOLOv8n@640 (50 эпох на MinneApple) и YOLO26s@1280 (40 эпох на MinneApple).
  Отдельного обучения на VFC в этой матрице нет.
- **Трекинг:** внешние `boxmot.BotSort`, `deep_sort_realtime.DeepSort` и
  `boxmot.ByteTrack`. Упрощённые `botsort_lite`/`deepsort_lite` предназначены
  только для отладки и не выдаются за эталонные алгоритмы.
- **Одинаковый вход:** версионированный NPZ-кэш детекций с SHA-256 источника и
  весов. Второй трекер не запускает YOLO повторно.
- **Метрики:** AP50/precision/recall для детекции; MOTA/IDF1/IDSW через
  `motmetrics` при постоянных GT ID; MAE/RMSE/MAPE для счёта при count GT.
- **Воспроизводимость:** `resolved_config.yaml`, `run_manifest.json`, версии
  пакетов, Git commit, время детекции и трекинга раздельно.
- **VFC:** 22 кадра маркеров задают 21 stretch. Reference-режим использует
  положение центра бокса, кадры маркеров и скорость; автоматическое деление
  stretch на мнимые деревья отключено.

## Датасеты и допустимые выводы

| Данные | Роль | Что нельзя утверждать |
|---|---|---|
| MinneApple train/val | обучение и оценка детекции на val | видео из train-кадров не является независимым тестом |
| MinneApple video | визуальная отладка | COCO annotation ID не дают MOTA/IDF1/IDSW и не являются track GT |
| T1, Zenodo 7384297 | отдельный MOT-бенчмарк | это не одно из 16 видео VFC |
| VideoFruitCounting | 16 видео, Kinect/ZED, счёт по 21 stretch | SFRAME не содержит GT числа яблок или деревьев |

Без ручного count GT на VFC можно сравнивать стабильность predictions и
фрагментацию, но нельзя выбирать комбинацию по «абсолютной точности счёта».

## Промежуточные результаты (09.11.2026)

Готовы: MinneApple val-детекция (3 детектора), полная MOT-матрица T1
(3 детектора × 2 трекера) и HSV-фаза VFC (16 видео × 2 трекера).
Сводный отчёт с таблицами, метриками и иллюстрациями: `report/`
(`REPORT.md` / `REPORT.pdf` / CSV-таблицы). Лучшая комбинация на текущих
данных: YOLOv8n + BoT-SORT (лучшие MOTA/IDF1 на T1); YOLO26s + BoT-SORT —
альтернатива с лучшей детекцией на MinneApple.

## Установка

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev,yolo,trackers,eval]"
```

Проверено в проектном `.venv` на Python 3.12. `setuptools<81` нужен текущему
`deep-sort-realtime` из-за `pkg_resources`.

## Быстрый запуск

```powershell
$env:PYTHONIOENCODING = "utf-8"
.venv\Scripts\python.exe -m applecounter demo --out outputs\demo_new
.venv\Scripts\python.exe -m pytest
```

Один прогон:

```powershell
.venv\Scripts\python.exe -m applecounter run `
  --video flight.mp4 --out outputs\flight_new `
  --set detector.name=yolo `
  --set detector.yolo.weights=weights/runs/minneapple50/weights/best.pt `
  --set tracker.name=botsort
```

По умолчанию CLI действительно загружает `configs/default.yaml`; `--set`
применяется после YAML.

## Исследовательская матрица

```powershell
# Только показать задачи, ничего не запускать
.venv\Scripts\python.exe scripts\run_matrix.py --dry-run

# Отдельный MOT-бенчмарк T1
.venv\Scripts\python.exe scripts\run_matrix.py --dataset t1

# 16 VFC × 3 детектора × 2 трекера; запускать после контрольных проверок
.venv\Scripts\python.exe scripts\run_matrix.py --dataset vfc
```

Матрица пишет только в `outputs/rework_matrix/` и
`outputs/rework_cache/`. Она не подставляет старые прогоны и считает задачу
завершённой только при валидном manifest с точно совпадающим конфигом.
`--only` фильтрует `dataset/detector/tracker/video`, `--max-frames` предназначен
для коротких smoke-тестов, `--frame-stride` меняет протокол и кэш.

Для основной матрицы: `counting.min_obs=1`, `counting.merge_dist=0`,
`trees.subdivide=false`, `yield.enabled=false`. Это сравнение сырых
подтверждённых треков, а не доказательство числа физических яблок.

## VFC и stretch

```powershell
.venv\Scripts\python.exe -m applecounter run `
  --video "data/raw/videofruit/Video Fruit Counting/Videos/Azure Kinect/210928_100624_k_r1_e_015_225_162.mp4" `
  --out outputs\vfc_control `
  --set video.rotate90=-1 `
  --set detector.name=yolo `
  --set detector.yolo.weights=weights/runs/minneapple26s1280/weights/best.pt `
  --set detector.yolo.imgsz=1280 `
  --set tracker.name=botsort `
  --set trees.method=segments `
  --set trees.assignment=vfc_reference `
  --set trees.segments_file="data/raw/videofruit/Video Fruit Counting/stretch_sframe.csv" `
  --set trees.subdivide=false
```

`n_double_count_events` лишь обнаруживает повторный `count_event` одного ID.
Он не измеряет повторный подсчёт одного физического плода разными ID.

Экспорт `tracks_frame.csv` в исходный формат авторов:

```powershell
.venv\Scripts\python.exe -m applecounter vfc-export `
  --run outputs\vfc_control `
  --segments "data/raw/videofruit/Video Fruit Counting/stretch_sframe.csv"
```

## Прогноз урожая

Физическая модель: `M = N × mean_mass / visibility_factor`. Без независимой
полевой калибровки это иллюстративная оценка, а не измеренная урожайность.
Регрессия использует `yield.regression_csv` с колонками `count,mass_kg` и
сохраняет отдельно in-sample и validation-метрики. Если задан
`yield.validation_csv`, выполняется внешний holdout; иначе — leave-one-out,
что не заменяет независимый контроль в другом саду/сезоне.

```powershell
.venv\Scripts\python.exe -m applecounter yield `
  --counts outputs\flight_new\counts.csv `
  --regression-csv field_train.csv `
  --set yield.validation_csv=field_holdout.csv
```

## Основные команды

| Команда | Назначение |
|---|---|
| `synth`, `demo` | синтетические данные и end-to-end проверка |
| `prepare`, `train`, `bench` | YOLO-датасет, обучение, val-детекция |
| `run`, `evaluate`, `report` | прогон, отдельная оценка, отчёт |
| `vfc-export` | CSV для внешнего VFC-инструмента |
| `yield` | калиброванная/иллюстративная оценка массы |

Полный экспериментальный протокол: `docs/experiment_protocol.md`.
Интерфейс трекеров: `docs/adding_a_tracker.md`.

## Лицензии

Код проекта — MIT. Внешние пакеты и веса сохраняют собственные лицензии;
каталог `external/VideoFruitCounting/` не входит в Git-коммиты проекта.
