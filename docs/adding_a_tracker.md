# Как подключить свой трекер (ByteTrack и не только)

Этап 2 проекта — сравнение трекеров: `ours` (собственный), `botsort`,
`deepsort` и **ByteTrack** (реализует другой соавтор). Все трекеры подключены
единым интерфейсом, поэтому новый трекер добавляется в 3 шага без изменений
в пайплайне.

## 1. Точка подключения

Фабрика `build_tracker(cfg)` в `src/applecounter/track/__init__.py`:

```python
def build_tracker(cfg):
    name = str(cfg.get("tracker.name", "ours")).lower()
    if name in ("ours", "custom", ""):
        return MultiObjectTracker(cfg)
    if name == "botsort":
        from .botsort import BotSortTracker
        return BotSortTracker(cfg)
    if name == "deepsort":
        from .deepsort import DeepSortTracker
        return DeepSortTracker(cfg)
    raise ValueError(f"неизвестный трекер: {name} (ожидается ours|botsort|deepsort)")
```

Ключ `tracker.name: ours` уже есть в `configs/default.yaml` и `DEFAULTS`.
Запуск: `applecounter run ... --set tracker.name=bytetrack`.

## 2. Контракт интерфейса

```python
class MyTracker:
    def __init__(self, cfg: Config): ...
    def update(self, frame_idx: int, detections: np.ndarray,
               image: Optional[np.ndarray] = None,
               offset: Tuple[float, float] = (0.0, 0.0)) -> List[TrackView]: ...
    @property
    def tracks(self) -> Dict[int, Track]: ...
```

| Параметр | Смысл |
|----------|-------|
| `frame_idx` | порядковый номер обработанного кадра (с 0) |
| `detections` | `np.float32 (N, 6)`: `x1, y1, x2, y2, conf, cls` — **в мировых координатах** (движение камеры уже скомпенсировано стабилизацией) |
| `image` | исходный кадр BGR (экран) — нужен только для признаков внешнего вида (re-ID) |
| `offset` | смещение мировой→экран: `screen_box = world_box + offset` |

Возвращаем `List[TrackView]` — по одной записи на каждый живой трек кадра:

| Поле | Смысл |
|------|-------|
| `track_id` | глобальный уникальный id |
| `box` | бокс **в мировых координатах** (предсказание, если детекции не было) |
| `conf`, `state`, `hits` | текущая надёжность/состояние (`TENTATIVE → CONFIRMED → LOST → REMOVED`) |
| `count_event` | `True` **ровно один раз** — когда трек подтверждён и начислен. Это гарантия от повторного счёта: счётчик `AppleCounter` при `True` создаёт запись плода, повторный `True` того же id считается ошибкой (`n_double_count_events`) |
| `updated` | `True`, если трек сопоставлен с реальной детекцией в этом кадре (без `updated` метрики MOTA и статистика наблюдений трека не считаются) |

Обязательные свойства после `update`:

- `tracks` — словарь `{track_id: Track}`; `counter.finalize` берёт из
  `Track.positions` (список `(frame, box)` в мировых координатах) историю для
  привязки к деревьям и слияния осколков;
- `n_created` — сколько треков создано за весь прогон (уходит в `summary.json`);
- удалённые треки (`REMOVED`) не должны оставаться в `tracks`.

Пороги читаются из `cfg["tracker"]`: `high_conf`, `low_conf`, `min_iou`,
`max_age`, `reid_max_age`, `n_init`, `use_appearance`, `appearance_gate`,
`reid_max_dist`, `velocity_damping`.

Семантика порогов (для сопоставимости сравнения):

- `conf >= high_conf` — «сильные» детекции (создают новые треки);
- `low_conf <= conf < high_conf` — «слабые»: в `ours` обновляют, но не
  создают; в `botsort` продлевают жизнь треков (двухэтапная ассоциация
  ByteTrack); в `deepsort` не участвуют (классический DeepSORT);
- `conf < low_conf` — игнорируются всеми.

## 3. Готовые строительные блоки

- `track/kalman.py` — `KalmanBoxFilter` (состояние `[u,v,w,h]+скорости`) и
  `KFTrackerBase`: общая обвязка (предсказание КФ, вызов `_associate`,
  bookkeeping, `TrackView`, `count_event`). **Наследуйте `KFTrackerBase` и
  переопределите только `_associate(active, dets, det_hists, pred_boxes,
  high_mask, low_mask) -> {tid: index_det}`** — так поступили `botsort.py` и
  `deepsort.py`;
- `track/tracker.py::_assign(cost, max_cost)` — венгерский алгоритм
  (scipy) с отсечкой по `max_cost`;
- `track/appearance.py` — `box_histogram(image, box)`, `cosine`,
  `appearance_cost` — HSV-гистограмма для re-ID.

### Замечание по лицензиям

Проект — MIT. Ultralytics (их YAML-трекеры) — AGPL-3.0, у VideoFruitCounting
(есть копия `tracking/bytetrack/` в `external/`) лицензии нет вовсе, поэтому
**код оттуда не копировать**: реализуйте ByteTrack по статье
(Zhang et al., 2022, arXiv:2110.06864) поверх `KFTrackerBase`. По сути это
две стадии ассоциации IoU из `botsort.py` (сильные → слабые) без третьей
re-ID-стадии, с КФ вместо наивного IoU-предсказания.

## 4. Тесты и сравнение

Добавьте в `tests/test_tracker.py` по образцу существующих:

```python
def test_bytetrack_static_counted_once():      # счёт ровно один раз
def test_bytetrack_gap_keeps_id():             # окклюзия не меняет id
def test_bytetrack_low_conf_fills_track():     # слабые детекции продлевают трек
```

и строку в `test_build_tracker_factory` (`build_tracker(Config({"tracker":
{"name": "bytetrack"}}))`).

Прогоны для таблицы сравнения (детектор одинаковый — меняется только трекер):

```powershell
# синтетика с эталоном счёта
applecounter run --video data/synthetic/synthetic_orchard.mp4 --gt data/synthetic/gt `
    --out outputs/synth_track_bytetrack --set tracker.name=bytetrack

# реальное видео MinneApple
applecounter run --video data/processed/minneapple_video/20150919_174730.mp4 `
    --gt data/processed/minneapple_video/gt --out outputs/minneapple_video_bytetrack `
    --set detector.name=yolo --set detector.yolo.weights=weights/runs/minneapple50/weights/best.pt `
    --set tracker.reid_max_dist=150 --set tracker.name=bytetrack
```

Метрики для таблицы: `summary.json` → `count_metrics` (счёт/MAE),
`tracking_metrics` (MOTA/IDSW), `tracker.created_tracks` (фрагментация).
Эталон по трекерам (см. README, раздел «Сравнение трекеров»): синтетика
MAE 0,17 у всех, IDSW 13 (ours) / 8 (botsort, deepsort); MinneApple
MOTA 0,215 (ours) / 0,516 (botsort) / 0,454 (deepsort).
