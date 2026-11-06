from .appearance import box_histogram, cosine  # noqa: F401
from .tracker import MultiObjectTracker, Track, TrackState, TrackView  # noqa: F401


def build_tracker(cfg):
    """Фабрика трекеров по конфигурации (tracker.name: ours|botsort|deepsort)."""
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
