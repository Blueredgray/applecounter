# Подключение и сравнение трекеров

## Уже подключённые реализации

| `tracker.name` | Реализация | Назначение |
|---|---|---|
| `botsort` | `boxmot.BotSort` | основной эксперимент: CMC + ReID |
| `deepsort` | `deep_sort_realtime.DeepSort` | основной эксперимент: CNN embeddings |
| `bytetrack` | `boxmot.ByteTrack` | этап соавтора, тот же интерфейс |
| `ours` | внутренний world-coordinate tracker | экспериментальная стабилизация |
| `botsort_lite`, `deepsort_lite` | упрощённые HSV baseline | только отладка и unit-тесты |

Нельзя называть `botsort_lite` и `deepsort_lite` эталонными BoT-SORT/DeepSORT
в статье или таблицах результатов.

## Контракт

```python
views = tracker.update(frame_idx, detections, image=frame, offset=offset)
```

- `detections`: `float32[N,6]`, колонки `x1,y1,x2,y2,conf,cls`.
- Для внешних BoT-SORT/ByteTrack/DeepSORT боксы находятся в screen coordinates.
  Они сами выполняют CMC, если он включён; внешняя ORB-стабилизация к ним не
  применяется дважды.
- `image`: текущий BGR-кадр, обязателен для DeepSORT и BoT-SORT с ReID.
- Каждый `TrackView` содержит `track_id`, `box`, `state`, фактическое число
  выданных наблюдений `hits`, `count_event` и `updated`.
- `count_event=True` допускается один раз при первом подтверждённом появлении ID.
  Это не доказывает отсутствие фрагментации физического яблока между разными ID.
- `tracks` хранит полную историю, `n_created` — число ID, принятых адаптером.

Фабрика находится в `src/applecounter/track/__init__.py`, адаптеры — в
`track/external.py`. Старые lite-реализации сохранены отдельно и не выбираются
именами `botsort`/`deepsort`.

## Условия честного сравнения

1. Одинаковый видеопоток, rotate/resize/stride и max_frames.
2. Один версионированный `detector.cache_file` для всех трекеров.
3. `counting.min_obs=1` для сравнения сырых подтверждённых треков; фильтры
   длины и post-hoc merge исследуются отдельно.
4. `counting.merge_dist=0` в основной матрице.
5. Для VFC — `trees.method=segments`, `trees.subdivide=false`,
   `trees.assignment=vfc_reference`.
6. MOTA/IDF1/IDSW рассчитывать только при постоянных GT ID (например, T1).
   COCO annotation ID в MinneApple video для этого не подходят.
7. Сохранять `resolved_config.yaml`, `run_manifest.json`, версии пакетов и хэши.

Запуск новой матрицы:

```powershell
.venv\Scripts\python.exe scripts\run_matrix.py --dry-run
.venv\Scripts\python.exe scripts\run_matrix.py --dataset t1
.venv\Scripts\python.exe scripts\run_matrix.py --dataset vfc
```

Старые каталоги `outputs/` матрицей не используются как готовые результаты.
