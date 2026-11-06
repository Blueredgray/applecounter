import numpy as np
import pytest

from applecounter.config import Config
from applecounter.track.tracker import MultiObjectTracker, TrackState


def _det(x1, y1, w=14, h=14, conf=0.9):
    return [x1, y1, x1 + w, y1 + h, conf, 0.0]


def test_static_object_counted_once():
    cfg = Config()
    tr = MultiObjectTracker(cfg)
    box = np.array([_det(100, 100)], dtype=np.float32)
    events = 0
    for f in range(10):
        views = tr.update(f, box)
        events += sum(1 for v in views if v.count_event)
    assert events == 1  # каждый объект начисляется ровно один раз
    assert len(tr.tracks) == 1
    track = next(iter(tr.tracks.values()))
    assert track.state == TrackState.CONFIRMED
    assert track.counted


def test_two_objects_two_counts():
    cfg = Config()
    tr = MultiObjectTracker(cfg)
    dets = np.array([_det(100, 100), _det(300, 100)], dtype=np.float32)
    events = 0
    for f in range(8):
        views = tr.update(f, dets)
        events += sum(1 for v in views if v.count_event)
    assert events == 2


def test_camera_shift_keeps_identity():
    """Мировые координаты статичны; offset (экран = мир + offset) меняется."""
    cfg = Config()
    tr = MultiObjectTracker(cfg)
    world = np.array([_det(500, 100)], dtype=np.float32)
    events = 0
    for f in range(12):
        offset = (-(f * 40.0), 0.0)  # камера едет вправо
        views = tr.update(f, world, offset=offset)
        events += sum(1 for v in views if v.count_event)
    assert events == 1
    assert len(tr.tracks) == 1
    assert len(tr.tracks) == 1


def test_reid_distance_gate_blocks_far_jump():
    """Трек не должен «переехать» на далёкий объект через re-ID."""
    cfg = Config({"tracker": {"n_init": 2, "reid_max_age": 60, "reid_max_dist": 60.0}})
    tr = MultiObjectTracker(cfg)
    # объект A живёт, затем пропадает (исчезает из кадра) надолго
    a = np.array([_det(100, 100)], dtype=np.float32)
    tr.update(0, a)
    tr.update(1, a)
    for f in range(2, 10):
        tr.update(f, np.zeros((0, 6), dtype=np.float32))
    # трек A уже LOST; на удалении 500 px появляется объект B — не должен получить id A
    b = np.array([_det(600, 100)], dtype=np.float32)
    views = tr.update(10, b)
    new_ids = [v.track_id for v in views if v.updated and v.track_id != 1]
    assert new_ids, "объект B должен получить новый id"
    track_a = tr.tracks.get(1)
    if track_a is not None:
        assert track_a.box[0] < 200, "трек A не должен перепривязаться к B"


def test_low_conf_detection_updates_but_not_creates():
    cfg = Config()
    tr = MultiObjectTracker(cfg)
    weak = np.array([_det(100, 100, conf=0.10)], dtype=np.float32)
    views = tr.update(0, weak)
    assert all(v.track_id >= 1 for v in views)  # новых треков из слабых детекций нет
    assert tr.n_created == 0


# ---------------------------------------------------------------------------
# BoT-SORT и DeepSORT (tracker.name)
# ---------------------------------------------------------------------------
def test_build_tracker_factory():
    from applecounter.track import build_tracker
    from applecounter.track.botsort import BotSortTracker
    from applecounter.track.deepsort import DeepSortTracker

    assert isinstance(build_tracker(Config()), MultiObjectTracker)
    assert isinstance(build_tracker(Config({"tracker": {"name": "botsort"}})), BotSortTracker)
    assert isinstance(build_tracker(Config({"tracker": {"name": "deepsort"}})), DeepSortTracker)
    try:
        build_tracker(Config({"tracker": {"name": "нет"}}))
        assert False, "должен быть ValueError"
    except ValueError:
        pass


def test_kalman_box_roundtrip():
    from applecounter.track.kalman import KalmanBoxFilter

    box = np.array([10.0, 20.0, 30.0, 40.0])
    z = KalmanBoxFilter.box_to_z(box)
    back = KalmanBoxFilter.z_to_box(z)
    assert np.allclose(box, back)


@pytest.mark.parametrize("name", ["botsort", "deepsort"])
def test_alt_trackers_static_counted_once(name):
    """BoT-SORT/DeepSORT: неподвижный объект начисляется один раз."""
    from applecounter.track import build_tracker

    cfg = Config({"tracker": {"name": name}})
    tr = build_tracker(cfg)
    box = np.array([_det(100, 100)], dtype=np.float32)
    events = 0
    for f in range(10):
        views = tr.update(f, box)
        events += sum(1 for v in views if v.count_event)
    assert events == 1
    assert len(tr.tracks) == 1
    assert next(iter(tr.tracks.values())).state == TrackState.CONFIRMED


@pytest.mark.parametrize("name", ["botsort", "deepsort"])
def test_alt_trackers_gap_keeps_id(name):
    """Краткий пропуск детекции (окклюзия) не должен менять id и считать заново."""
    from applecounter.track import build_tracker

    cfg = Config({"tracker": {"name": name}})
    tr = build_tracker(cfg)
    box = np.array([_det(100, 100)], dtype=np.float32)
    for f in range(3):
        tr.update(f, box)
    for f in range(3, 6):  # окклюзия
        tr.update(f, np.zeros((0, 6), dtype=np.float32))
    events = 0
    for f in range(6, 9):
        views = tr.update(f, box)
        events += sum(1 for v in views if v.count_event)
        assert all(v.track_id == 1 for v in views), "id должен сохраниться"
    assert events == 0, "повторного начисления быть не должно"
    assert len(tr.tracks) == 1


def test_botsort_low_conf_fills_track():
    """BoT-SORT: низконадёжные детекции продолжают жизнь трека (ByteTrack-этап)."""
    from applecounter.track.botsort import BotSortTracker

    cfg = Config({"tracker": {"name": "botsort", "max_age": 25}})
    tr = BotSortTracker(cfg)
    strong = np.array([_det(100, 100, conf=0.9)], dtype=np.float32)
    for f in range(3):
        tr.update(f, strong)
    weak = np.array([_det(100, 100, conf=0.2)], dtype=np.float32)  # 0.15 <= conf < 0.45
    views = tr.update(3, weak)
    updated = [v for v in views if v.updated and v.state == TrackState.CONFIRMED]
    assert updated, "слабая детекция должна продлить существующий трек"
    assert tr.n_created == 1  # новых треков из слабых не появилось


def test_deepsort_ignores_low_conf():
    """DeepSORT: классически низконадёжные детекции не участвуют в ассоциации."""
    from applecounter.track.deepsort import DeepSortTracker

    cfg = Config({"tracker": {"name": "deepsort"}})
    tr = DeepSortTracker(cfg)
    strong = np.array([_det(100, 100, conf=0.9)], dtype=np.float32)
    for f in range(3):
        tr.update(f, strong)
    weak = np.array([_det(100, 100, conf=0.2)], dtype=np.float32)
    views = tr.update(3, weak)
    updated = [v for v in views if v.updated]
    assert not updated, "слабая детекция не должна обновлять трек DeepSORT"
