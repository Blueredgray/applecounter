from .appearance import box_histogram, cosine  # noqa: F401
from .tracker import MultiObjectTracker, Track, TrackState, TrackView  # noqa: F401


def build_tracker(cfg):
    """Build a tracker without confusing lightweight baselines with reference methods."""
    name = str(cfg.get("tracker.name", "ours")).lower()
    if name in ("ours", "custom", ""):
        return MultiObjectTracker(cfg)
    if name in ("botsort_lite", "deepsort_lite"):
        if name == "botsort_lite":
            from .botsort import BotSortTracker

            return BotSortTracker(cfg)
        from .deepsort import DeepSortTracker

        return DeepSortTracker(cfg)
    if name in ("botsort", "bytetrack"):
        from .external import BoxMotTracker

        return BoxMotTracker(cfg, name)
    if name == "deepsort":
        from .external import DeepSortRealtimeTracker

        return DeepSortRealtimeTracker(cfg)
    raise ValueError(
        f"неизвестный трекер: {name} "
        "(ours|botsort|bytetrack|deepsort|botsort_lite|deepsort_lite)"
    )
