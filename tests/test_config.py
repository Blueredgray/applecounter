import yaml

from applecounter.config import Config, DEFAULTS, deep_merge, load_config


def test_defaults_present():
    cfg = Config()
    assert cfg.get("detector.name") == "hsv"
    assert cfg.get("tracker.n_init") == 2
    assert cfg.get("counting.merge_dist") > 0


def test_get_missing_returns_default():
    cfg = Config()
    assert cfg.get("нет.такого", 42) == 42


def test_set_and_get_dotted():
    cfg = Config()
    cfg.set("tracker.n_init", 5)
    assert cfg.get("tracker.n_init") == 5


def test_deep_merge():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    out = deep_merge(base, {"a": {"b": 10}, "e": 4})
    assert out == {"a": {"b": 10, "c": 2}, "d": 3, "e": 4}
    assert base["a"]["b"] == 1  # без мутации


def test_load_yaml_and_overrides(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"tracker": {"n_init": 4}, "trees": {"n_trees": 5}}),
                    encoding="utf-8")
    cfg = load_config(str(path), overrides=["tracker.max_age=7", "detector.name=yolo",
                                            "trees.n_trees=null"])
    assert cfg.get("tracker.n_init") == 4
    assert cfg.get("tracker.max_age") == 7
    assert cfg.get("detector.name") == "yolo"
    assert cfg.get("trees.n_trees") is None


def test_load_missing_file(tmp_path):
    try:
        load_config(str(tmp_path / "нет.yaml"))
        assert False, "должна быть ошибка"
    except FileNotFoundError:
        pass


def test_defaults_not_mutated_by_config():
    cfg = Config()
    cfg.set("tracker.n_init", 99)
    assert DEFAULTS["tracker"]["n_init"] == 2
