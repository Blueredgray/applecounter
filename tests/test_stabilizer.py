import cv2
import numpy as np

from applecounter.config import Config
from applecounter.trees.stabilize import Stabilizer


def _texture(seed=0, w=320, h=240):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img


def test_stabilizer_tracks_shift():
    base = _texture()
    cfg = Config({"stabilizer": {"enabled": True, "min_matches": 8}})
    stab = Stabilizer(cfg)
    dx = 6.0
    offsets = []
    for f in range(8):
        # сдвигаем базу влево => содержимое кадра смещается
        M = np.float32([[1, 0, -dx * f], [0, 1, 0]])
        frame = cv2.warpAffine(base, M, (base.shape[1], base.shape[0]), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
        off = stab.step(frame)
        offsets.append(off.copy())
    # накопленное смещение по x должно монотонно убывать (содержимое едет влево)
    xs = [o[0] for o in offsets]
    assert xs[-1] < xs[0]
    assert abs(xs[-1] - (-dx * 7)) < 6.0, f"накоплено {xs[-1]:.1f}, ожидалось ~{-dx * 7}"
    assert stab.n_estimated >= 5


def test_stabilizer_disabled():
    cfg = Config({"stabilizer": {"enabled": False}})
    stab = Stabilizer(cfg)
    out = stab.step(np.zeros((100, 100), dtype=np.uint8))
    np.testing.assert_allclose(out, [0, 0])
    assert stab.n_estimated == 0


def test_path_length():
    cfg = Config()
    stab = Stabilizer(cfg)
    for f in range(5):
        img = _texture(seed=f)
        stab.step(img)
    assert stab.path_length >= 0.0
    assert stab.n_estimated + stab.n_failed == 4  # первый кадр без пары
