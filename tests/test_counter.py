import numpy as np

from applecounter.counting.counter import AppleCounter, TrackRecord
from applecounter.trees.segment import TreeMap


def _view(tid, box, conf=0.9, count_event=False, updated=True, state=None, hits=2):
    from applecounter.track.tracker import TrackState, TrackView

    return TrackView(
        track_id=tid, box=tuple(box), conf=conf,
        state=state or TrackState.CONFIRMED, hits=hits,
        count_event=count_event, updated=updated,
    )


def test_count_once_and_no_double_events():
    c = AppleCounter(min_obs=2)
    box = [10, 10, 24, 24]
    c.update(0, [_view(1, box, count_event=True)])
    c.update(1, [_view(1, box, count_event=False)])
    c.update(2, [_view(1, box, count_event=True)])  # «повторный» count_event — не должен считаться
    res = c.finalize(None, None, merge_dist=0)
    assert res.total == 1
    assert c.n_double_count_events == 1  # зафиксирован, но не увеличил счёт


def test_min_obs_filter():
    c = AppleCounter(min_obs=3)
    c.update(0, [_view(1, [0, 0, 10, 10], count_event=True, hits=1)])
    c.update(1, [_view(1, [0, 0, 10, 10], hits=1)])
    res = c.finalize(None, None, merge_dist=0)
    assert res.total == 0  # два фактических наблюдения вместо трёх


def test_no_merge_of_simultaneous_tracks():
    c = AppleCounter(min_obs=2)
    c.update(0, [_view(1, [100, 100, 114, 114], count_event=True)])
    c.update(1, [_view(1, [100, 100, 114, 114]),
                 _view(2, [104, 102, 116, 114], count_event=True)])
    c.update(2, [_view(2, [104, 102, 116, 114])])
    res = c.finalize(None, None, merge_dist=14.0)
    assert res.total == 2


def test_merge_fragments_same_apple():
    """Два трека одного плода (близкие центры) сливаются в один подсчёт."""
    c = AppleCounter(min_obs=2)
    c.update(0, [_view(1, [100, 100, 114, 114], count_event=True)])
    c.update(1, [_view(1, [100, 100, 114, 114])])
    c.update(2, [_view(2, [104, 102, 116, 114], count_event=True)])  # осколок рядом
    c.update(3, [_view(2, [104, 102, 116, 114])])
    res = c.finalize(None, None, merge_dist=14.0)
    assert res.total == 1


def test_no_merge_of_distinct_apples():
    c = AppleCounter(min_obs=2)
    c.update(0, [_view(1, [100, 100, 114, 114], count_event=True),
                 _view(2, [140, 100, 154, 114], count_event=True)])
    c.update(1, [_view(1, [100, 100, 114, 114]), _view(2, [140, 100, 154, 114])])
    res = c.finalize(None, None, merge_dist=14.0)
    assert res.total == 2


def test_per_tree_assignment_and_per_frame():
    c = AppleCounter(min_obs=2)
    # дерево 1: x < 100, дерево 2: x >= 100 (граница)
    tree_map = TreeMap(method="axis", origin=np.zeros(2),
                       direction=np.array([1.0, 0.0]), boundaries=np.array([100.0]))
    c.update(0, [_view(1, [10, 10, 24, 24], count_event=True),
                 _view(2, [200, 10, 214, 24], count_event=True)])
    c.update(1, [_view(1, [10, 10, 24, 24]), _view(2, [200, 10, 214, 24])])
    res = c.finalize(tree_map, None, merge_dist=14.0)
    assert res.total == 2
    assert res.per_tree == {1: 1, 2: 1}
    assert res.unassigned == 0
    assert res.per_frame_ids == [2, 2]
    rows = res.as_rows()
    assert {r["tree_id"] for r in rows} == {1, 2}


def test_records_sorted_by_first_frame():
    c = AppleCounter(min_obs=2)
    c.update(0, [_view(5, [200, 10, 214, 24], count_event=True)])
    c.update(1, [_view(5, [200, 10, 214, 24])])
    c.update(2, [_view(7, [10, 10, 24, 24], count_event=True)])
    c.update(3, [_view(7, [10, 10, 24, 24])])
    res = c.finalize(None, None, merge_dist=0)
    assert [r.track_id for r in res.records] == [5, 7]
