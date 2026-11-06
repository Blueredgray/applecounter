# Итоговый план исследовательских прогонов

Дата актуализации: 2026-11-06.

## Что уже подтверждено

1. Исходная реализация сохранена отдельным коммитом `f14b87e`.
2. После переработки полный набор тестов: **111 passed**.
3. Короткий VFC-прогон YOLOv8n + BoT-SORT, 12 кадров:
   - `cache=created`;
   - валидный `run_manifest.json`;
   - единица счёта `stretch`;
   - `trees.assignment=vfc_reference`;
   - `yield=None`.
4. Тот же фильтр с DeepSORT:
   - `cache=loaded`;
   - SHA-256 кэша совпал с BoT-SORT;
   - валидный манифест.
5. `retree` на реальном коротком VFC-прогоне сохранил `vfc_reference`, не вернулся к `median_frame` и не оставил включённый прогноз урожая.
6. Полный сухой план VFC формирует 96 задач: 16 видео × 3 детектора × 2 локальных трекера.
7. Локальная матрица намеренно включает BoT-SORT и DeepSORT. ByteTrack подключён через общий интерфейс и оставлен партнёрскому этапу.
8. На текущей машине `pdflatex` не найден, поэтому PDF пока нельзя собрать здесь.

## 0. Предварительные условия

Все команды ниже предполагают PowerShell и корень `G:\opencode\soft\applecounter3`:

```powershell
Set-Location "G:\opencode\soft\applecounter3"
$env:PYTHONPATH="G:\opencode\soft\applecounter3\src"
$env:PYTHONIOENCODING="utf-8"
```

Проверить до запусков:

- существуют веса:
  - `weights/runs/minneapple50/weights/best.pt`;
  - `weights/runs/minneapple26s1280/weights/best.pt`;
- VFC содержит ровно 16 MP4 и `stretch_sframe.csv`;
- T1 содержит `img1` и `gt/gt.txt`;
- для MinneApple подготовлен YOLO-датасет с `data.yaml`.

Сейчас `data/processed/**/data.yaml` не найден. Если MinneApple ещё не конвертирован, сначала выполнить загрузку и `applecounter prepare`; до появления `data.yaml` бенчмарк детекции на val запускать бессмысленно.

## 1. Детекция на MinneApple val

Запускать три отдельных варианта, чтобы у каждого были зафиксированные веса и размер входа:

```powershell
.\.venv\Scripts\python.exe -m applecounter bench `
  --data data\processed\minneapple `
  --detector hsv `
  --out outputs\minneapple_detection\bench_hsv.json

.\.venv\Scripts\python.exe -m applecounter bench `
  --data data\processed\minneapple `
  --detector yolo `
  --yolo-weights weights\runs\minneapple50\weights\best.pt `
  --set detector.yolo.imgsz=640 `
  --out outputs\minneapple_detection\bench_yolo8n_640.json

.\.venv\Scripts\python.exe -m applecounter bench `
  --data data\processed\minneapple `
  --detector yolo `
  --yolo-weights weights\runs\minneapple26s1280\weights\best.pt `
  --set detector.yolo.imgsz=1280 `
  --out outputs\minneapple_detection\bench_yolo26s_1280.json
```

Зафиксировать AP@0.50, P, R, F1 и count-MAE. Эти значения могут отличаться от значений стандартного валидатора Ultralytics — для статьи нужно явно указать свой протокол и не смешивать две методики.

## 2. Полная MOT-матрица на T1

Разделять запуски по детекторам, чтобы при обрыве был безопасный возобновляемый этап:

```powershell
.\.venv\Scripts\python.exe scripts\run_matrix.py --dataset t1 --only "/hsv/"
.\.venv\Scripts\python.exe scripts\run_matrix.py --dataset t1 --only "/yolo8n/"
.\.venv\Scripts\python.exe scripts\run_matrix.py --dataset t1 --only "/yolo26s/"
```

Ожидание: 6 валидных прогонов = 3 детектора × 2 трекера. Для каждого детектора BoT-SORT и DeepSORT должны использовать один и тот же кэш. Метрики T1: MOTA, IDF1, IDSW, FP/FN. Эти метрики оценивают траектории, а не доказывают правильный итоговый счёт яблок.

## 3. Полная VFC-матрица

После контрольного короткого прогона запускать полные видео, разделяя этапы по детектору:

```powershell
.\.venv\Scripts\python.exe scripts\run_matrix.py --dataset vfc --only "/hsv/"
.\.venv\Scripts\python.exe scripts\run_matrix.py --dataset vfc --only "/yolo8n/"
.\.venv\Scripts\python.exe scripts\run_matrix.py --dataset vfc --only "/yolo26s/"
```

Ожидание: 96 валидных прогонов. Скрипт пропускает только те задачи, у которых есть завершённый манифест и точное совпадение конфигурации. После обрыва питания или перезагрузки повторный запуск без `--rerun` продолжает незавершённые задачи. Не использовать `--rerun`, если нужно сохранить уже валидные результаты.

В `outputs/rework_matrix/matrix_results.csv` должны остаться только валидные прогоны. Следить за журналом `outputs/rework_matrix.log` и за строками `ОШИБКА`.

## 4. Ручной count GT для VFC — обязательный этап

SFRAME содержит кадры маркеров stretch, но не числа яблок. Без ручного count GT нельзя:

- выбрать лучший детектор или трекер по точности счёта VFC;
- заявить ошибку подсчёта в процентах;
- сравнить BoT-SORT и DeepSORT как методы именно подсчёта.

Порядок:

1. До просмотра предсказаний зафиксировать разбиение 16 видео на validation и control. Разбиение должно быть сбалансировано по камерам, рядам и сторонам.
2. Создать `data/vfc_count_gt.csv` с минимальными колонками:
   `video,stretch,count,subset,annotator,status,notes`.
3. Для каждого выбранного видео разметчик считает уникальные яблоки по каждому stretch по полному видео, не по одному кадру и не по SFRAME.
4. Не копировать наши `tree_id` в GT. Идентификатор stretch берётся только из SFRAME.
5. Желательно разметить control-набор дважды двумя разметчиками и разрешить расхождения до оценки.
6. После заморозки GT не менять его под результат моделей.

Минимально для публикации: все 21 stretch каждого control-видео. Если это невозможно, уменьшать число видео, а не количество stretch внутри случайно выбранных видео; ограничение обязательно указать в статье.

## 5. Оценка счёта и выбор комбинации

1. Добавить отдельный evaluation-скрипт, который читает `outputs/rework_matrix/**/counts.csv` и замороженный `data/vfc_count_gt.csv`.
2. На validation-видео можно выбирать параметры, например `min_obs` или допустимость ablation merge.
3. Финальная таблица строится только на control-видео.
4. Метрики: MAE, RMSE, bias, MAPE и относительная ошибка итога по видео.
5. Не выбирать метод по числу созданных треков, MOTA или сумме `total_apples` без count GT.
6. Отдельно сохранить результат с `min_obs>1` и `merge_dist>0` как ablation, не подменяя им основную матрицу.

## 6. Прогноз урожая

1. До полевой калибровки физический прогноз публиковать только как иллюстративный (`estimate_type=illustrative_unless_field_calibrated`).
2. Для валидированной регрессии собрать независимый CSV `tree_id,count,mass_kg` по взвешенным контрольным деревьям.
3. Обучение и holdout должны быть разделены:
   - `yield.regression_csv` — train;
   - `yield.validation_csv` — независимый holdout.
4. Минимум для устойчивой регрессии — существенно больше формального минимума в 3 наблюдения; желательно не менее 20–30 деревьев с разным числом плодов.
5. Если масса контрольных деревьев не собрана, из статьи убрать выводы о точности прогноза кг.

## 7. Заполнение статьи

1. Таблицы `paper/article.tex` заполнять только из новых валидных прогонов и замороженного count GT.
2. Не переносить результаты старой реализации и smoke-тесты 12 кадров.
3. Отдельно указать:
   - MinneApple использован для детекции, не для видеотрекинга;
   - T1 не является VideoFruitCounting;
   - VFC оценивается по stretch, а не по автоматически найденным деревьям;
   - 12-кадровые прогоны — только инфраструктурная проверка.
4. Собрать PDF на машине с установленным `pdflatex`. На текущей машине компилятор не обнаружен.

## 8. Финальная фиксация результата

После завершения матрицы:

1. Проверить количество валидных строк в `matrix_results.csv`.
2. Проверить совпадение SHA-256 кэша между трекерами внутри каждого видео и детектора.
3. Проверить, что старые `outputs` не участвовали в таблицах.
4. Зафиксировать новый Git-коммит с кодом, тестами, протоколом и заполненной статьёй.
5. В commit не включать `data/`, веса, выходы полных прогонов, `models/` и вложенный `external/VideoFruitCounting/`.
6. Отдельно сохранить архив результатов вместе с `matrix_results.csv`, манифестами и SHA-256 входных данных.

## Осознанно не входит в текущий этап

- Компенсация движения камеры и восстановление ID после длительных пропусков — третий этап.
- Автоматическое дробление stretch на отдельные деревья — не использовать для научных выводов.
- ByteTrack-прогоны — партнёрский этап поверх того же кэша и интерфейса.
- Валидация абсолютной массы урожая без полевого взвешивания.
