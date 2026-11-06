import cv2
import numpy as np
import yaml

from applecounter.data.convert import (
    _split_key,
    convert_acfr,
    convert_minneapple,
    mask_to_boxes,
)


def _write_image(path, h=60, w=80):
    img = np.full((h, w, 3), 120, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_mask_to_boxes_two_blobs():
    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    mask[30:45, 40:55] = 255
    boxes = mask_to_boxes(mask, min_area=10)
    assert len(boxes) == 2
    x1s = sorted(b[0] for b in boxes)
    assert x1s == [10, 40]


def test_mask_to_boxes_inverted():
    """Если маска инвертирована (фон 255) — яблоками считается меньшинство."""
    mask = np.full((60, 80), 255, dtype=np.uint8)
    mask[10:20, 10:20] = 0
    boxes = mask_to_boxes(mask, min_area=10)
    assert len(boxes) == 1


def test_mask_to_boxes_empty():
    assert mask_to_boxes(np.zeros((30, 30), dtype=np.uint8)) == []


def test_mask_to_boxes_three_channel():
    """Маски ACFR: 3 канала, значение 128, яблоко не в первом канале."""
    mask = np.zeros((60, 80, 3), dtype=np.uint8)
    mask[10:20, 10:20, 1] = 128
    boxes = mask_to_boxes(mask)
    assert len(boxes) == 1
    assert boxes[0][0] == 10 and boxes[0][1] == 10


def _fake_minneapple(root):
    for split in ("train", "val"):
        (root / split / "imgs").mkdir(parents=True)
        (root / split / "masks").mkdir(parents=True)
        for i in range(4):
            _write_image(root / split / "imgs" / f"img_{i}.jpg")
            mask = np.zeros((60, 80), dtype=np.uint8)
            mask[10 + i:20 + i, 10:20] = 255
            cv2.imwrite(str(root / split / "masks" / f"img_{i}.png"), mask)


def test_convert_minneapple(tmp_path):
    src = tmp_path / "minneapple"
    _fake_minneapple(src)
    out = tmp_path / "out"
    result = convert_minneapple(str(src), str(out), val_ratio=0.2)
    assert result["train"] + result["val"] == 8
    labels = list((out / "labels" / "train").glob("*.txt")) + list((out / "labels" / "val").glob("*.txt"))
    assert len(labels) == 8
    text = labels[0].read_text(encoding="utf-8").strip().split()
    assert text[0] == "0" and len(text) == 5
    data = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
    assert data["nc"] == 1 and data["names"][0] == "apple"


def _fake_minneapple_coco(root):
    import json

    for split in ("train", "test"):
        (root / split / "images").mkdir(parents=True)
    _write_image(root / "train" / "images" / "a.png", h=200, w=100)
    (root / "train" / "train.json").write_text(json.dumps({
        "images": [{"id": 1, "file_name": "a.png", "width": 100, "height": 200}],
        "annotations": [{"id": 1, "image_id": 1, "bbox": [10, 20, 30, 40],
                         "area": 1200, "iscrowd": 0}],
        "categories": [{"id": 1, "name": "apple"}],
    }), encoding="utf-8")
    _write_image(root / "test" / "images" / "b.png", h=200, w=100)
    (root / "test" / "test.json").write_text(json.dumps({
        "images": [{"id": 1, "file_name": "b.png", "width": 100, "height": 200}],
        "annotations": [{"id": 2, "image_id": 1, "bbox": [0, 0, 50, 50],
                         "area": 2500, "iscrowd": 0}],
        "categories": [{"id": 1, "name": "apple"}],
    }), encoding="utf-8")


def test_convert_minneapple_coco(tmp_path):
    """COCO-раскладка MinneApple (HF): bbox [x,y,w,h], test -> val."""
    src = tmp_path / "minneapple"
    _fake_minneapple_coco(src)
    out = tmp_path / "out_coco"
    result = convert_minneapple(str(src), str(out))
    assert result["format"] == "coco"
    assert result["train"] == 1 and result["val"] == 1 and result["boxes"] == 2
    fields = (out / "labels" / "train" / "a.txt").read_text(encoding="utf-8").split()
    assert fields[0] == "0"
    assert abs(float(fields[1]) - 0.25) < 1e-6   # (10+15)/100
    assert abs(float(fields[2]) - 0.20) < 1e-6   # (20+20)/200
    assert abs(float(fields[3]) - 0.30) < 1e-6
    assert abs(float(fields[4]) - 0.20) < 1e-6
    assert (out / "images" / "val" / "b.png").exists()


def test_convert_minneapple_coco_val_carved(tmp_path):
    """Если eval-сплитов нет — val выдерживается из train."""
    src = tmp_path / "minneapple"
    (src / "train" / "images").mkdir(parents=True)
    for i in range(10):
        _write_image(src / "train" / "images" / f"img_{i}.png")
    import json
    (src / "train" / "train.json").write_text(json.dumps({
        "images": [{"id": i, "file_name": f"img_{i}.png", "width": 80, "height": 60}
                   for i in range(10)],
        "annotations": [{"id": i, "image_id": i, "bbox": [10, 10, 10, 10], "area": 100}
                        for i in range(10)],
        "categories": [{"id": 1, "name": "apple"}],
    }), encoding="utf-8")
    out = tmp_path / "out"
    result = convert_minneapple(str(src), str(out))
    assert result["val"] > 0 and result["train"] > 0
    assert result["train"] + result["val"] == 10


def _fake_acfr(root):
    apples = root / "acfr-fruit-dataset" / "apples"
    (apples / "images").mkdir(parents=True)
    (apples / "segmentations").mkdir(parents=True)
    (apples / "sets").mkdir(parents=True)
    for i in range(6):
        _write_image(apples / "images" / f"f_{i:03d}.png")
        mask = np.zeros((60, 80), dtype=np.uint8)
        mask[5:25, 5:25] = 1  # инстанс с id=1
        cv2.imwrite(str(apples / "segmentations" / f"f_{i:03d}.png"), mask)
    (apples / "sets" / "train.txt").write_text("f_000\nf_001\nf_002\nf_003\n", encoding="utf-8")
    (apples / "sets" / "val.txt").write_text("f_004\nf_005\n", encoding="utf-8")


def test_convert_acfr(tmp_path):
    src = tmp_path / "acfr"
    _fake_acfr(src)
    out = tmp_path / "out_acfr"
    result = convert_acfr(str(src), str(out))
    assert result["train"] == 4 and result["val"] == 2
    train_labels = list((out / "labels" / "train").glob("*.txt"))
    val_labels = list((out / "labels" / "val").glob("*.txt"))
    assert len(train_labels) == 4 and len(val_labels) == 2
    assert train_labels[0].read_text(encoding="utf-8").strip().split("\n")[0].count(" ") == 4  # 1 бокс
    data = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
    assert data["train"] == "images/train"


def test_split_key_dotted_names():
    """Имена ACFR содержат точки: расширение снимается только у картинок."""
    assert _split_key("20130320T004348.182606.Cam6_54") == "20130320T004348.182606.Cam6_54"
    assert _split_key("sets/20130320T004348.182606.Cam6_54.png") == "20130320T004348.182606.Cam6_54"


def test_convert_acfr_real_naming(tmp_path):
    """Реальные имена ACFR: точки в stem, маски <stem>_L.png, сплит из sets/."""
    apples = tmp_path / "acfr-fruit-dataset" / "apples"
    (apples / "images").mkdir(parents=True)
    (apples / "segmentations").mkdir(parents=True)
    (apples / "sets").mkdir(parents=True)
    stems = ["20130320T004348.182606.Cam6_54", "20130320T004349.135000.Cam6_11",
             "20130320T004352.754200.Cam6_23"]
    for stem in stems:
        _write_image(apples / "images" / f"{stem}.png")
        mask = np.zeros((60, 80), dtype=np.uint8)
        mask[5:25, 5:25] = 1
        cv2.imwrite(str(apples / "segmentations" / f"{stem}_L.png"), mask)
    (apples / "sets" / "train.txt").write_text("\n".join(stems[:2]), encoding="utf-8")
    (apples / "sets" / "val.txt").write_text(stems[2], encoding="utf-8")
    (apples / "sets" / "train_val.txt").write_text("\n".join(stems), encoding="utf-8")

    out = tmp_path / "out"
    result = convert_acfr(str(tmp_path), str(out))
    assert result["train"] == 2 and result["val"] == 1  # train_val не засчитан дважды
    assert len(list((out / "labels" / "val").glob("*.txt"))) == 1


def test_convert_acfr_annotations_csv(tmp_path):
    """Главная логика ACFR: боксы из annotations/*.csv, даже если маска пуста."""
    apples = tmp_path / "acfr-fruit-dataset" / "apples"
    for sub in ("images", "segmentations", "annotations", "sets"):
        (apples / sub).mkdir(parents=True)
    stems = ["20130320T004348.182606.Cam6_54", "20130320T004351.992184.Cam6_61"]
    for stem in stems:
        _write_image(apples / "images" / f"{stem}.png")
        cv2.imwrite(str(apples / "segmentations" / f"{stem}_L.png"),
                    np.zeros((60, 80, 3), dtype=np.uint8))  # маска пуста!
    # image0: два яблока; image1: только заголовок (annotated как фон)
    (apples / "annotations" / f"{stems[0]}.csv").write_text(
        "#item,c-x,c-y,radius,label\n"
        "0,20.0,30.0,10.0,1\n"
        "1,50.0,40.0,8.0,1\n"
        "2,5.0,5.0,3.0,0\n", encoding="utf-8")
    (apples / "annotations" / f"{stems[1]}.csv").write_text(
        "#item,c-x,c-y,radius,label\n", encoding="utf-8")
    (apples / "sets" / "train.txt").write_text(stems[0], encoding="utf-8")
    (apples / "sets" / "val.txt").write_text(stems[1], encoding="utf-8")

    out = tmp_path / "out"
    result = convert_acfr(str(tmp_path), str(out))
    assert result["train"] == 1 and result["val"] == 1
    lbl = (out / "labels" / "train" / f"{stems[0]}.txt").read_text(encoding="utf-8").strip()
    lines = [l for l in lbl.splitlines() if l.strip()]
    assert len(lines) == 2  # две метки label=1, фон label=0 не взят
    fields = lines[0].split()
    assert len(fields) == 5 and fields[0] == "0"
