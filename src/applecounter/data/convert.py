"""Конвертация датасетов в формат YOLO для дообучения детектора.

Поддерживается структура с масками (MinneApple, ACFR): боксы получаются из
связных компонент маски; сплиты train/val берутся из штатных каталогов/файлов.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from ..utils.logging import get_logger

logger = get_logger()

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASSES = ["apple"]


# ---------------------------------------------------------------------------
# Маска -> боксы
# ---------------------------------------------------------------------------
def mask_to_boxes(mask: np.ndarray, min_area: int = 20) -> List[Tuple[float, float, float, float]]:
    """Связные компоненты маски -> боксы (x1, y1, x2, y2) в пикселях."""
    if mask.ndim == 3:
        # объединяем каналы: в ACFR маски 3-канальные со значением 128,
        # яблоки могут быть только в одном канале
        mask = mask.max(axis=2)
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return []
    # «яблоки» — меньшинство пикселей: если маска инвертирована — разворачиваем
    if binary.mean() > 0.5:
        binary = 1 - binary
        if binary.sum() == 0:
            return []

    # многоклассовые маски (id инстансов) обрабатываем по значениям
    values = [v for v in np.unique(mask) if v != 0]
    boxes: List[Tuple[float, float, float, float]] = []
    if len(values) > 2 and len(values) <= 512:
        for value in values:
            comp = ((mask == value).astype(np.uint8))
            n, labels, stats, _ = cv2.connectedComponentsWithStats(comp, connectivity=8)
            for i in range(1, n):
                x, y, w, h, area = stats[i]
                if area >= min_area:
                    boxes.append((float(x), float(y), float(x + w), float(y + h)))
    else:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area >= min_area:
                boxes.append((float(x), float(y), float(x + w), float(y + h)))
    return boxes


def _write_yolo_label(label_path: Path, boxes: Sequence[Tuple[float, float, float, float]],
                      img_w: int, img_h: int) -> None:
    lines = []
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2.0 / max(img_w, 1)
        cy = (y1 + y2) / 2.0 / max(img_h, 1)
        w = (x2 - x1) / max(img_w, 1)
        h = (y2 - y1) / max(img_h, 1)
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        w, h = min(max(w, 0.0), 1.0), min(max(h, 0.0), 1.0)
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_data_yaml(root: Path, n_train: int, n_val: int) -> Path:
    data = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CLASSES)},
        "nc": len(CLASSES),
    }
    path = root / "data.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    logger.info("data.yaml: train=%d, val=%d", n_train, n_val)
    return path


def _split_key(line: str) -> str:
    """Ключ сплита: имя файла без каталога и без расширения изображения.

    Важно: у ACFR имена вида ``20130320T004348.182606.Cam6_54`` содержат
    точки — ``Path.stem`` здесь обрезал бы их по последней точке, поэтому
    расширение снимается только для известных расширений изображений.
    """
    name = Path(line.replace("\\", "/")).name
    suffix = Path(name).suffix.lower()
    if suffix in IMG_EXTS:
        name = name[: len(name) - len(suffix)]
    return name


def _parse_acfr_annotations(ann_dir: Path, stem: str) -> Optional[List[Tuple[float, float, float, float]]]:
    """Annotations ACFR: CSV ``#item,c-x,c-y,radius,label`` -> боксы.

    Возвращает ``None``, если файла аннотаций нет (тогда берём маску),
    и список (возможно пустой), если файл есть — это главный источник истины
    (часть масок ACFR пуста даже при наличии аннотаций).
    """
    path = ann_dir / f"{stem}.csv"
    if not path.exists():
        return None
    boxes: List[Tuple[float, float, float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            cx, cy, radius = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue
        if parts[4] not in ("", "1"):  # 0 — фон, 1 — яблоко
            continue
        boxes.append((cx - radius, cy - radius, cx + radius, cy + radius))
    return boxes


def _find_dirs(root: Path, names: Sequence[str]) -> Optional[Path]:
    for path in [root, *root.rglob("*")]:
        if path.is_dir() and path.name.lower() in {n.lower() for n in names}:
            return path
    return None


def _pairs_from_dirs(images_dir: Path, masks_dir: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        stem = img.stem
        candidates = [masks_dir / f"{stem}{ext}" for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff")]
        mask = next((c for c in candidates if c.exists()), None)
        if mask is None:
            # попытка «мягкого» совпадения (MinneApple использует mask_ префикс и т.п.)
            matches = [m for m in masks_dir.iterdir() if m.suffix.lower() in IMG_EXTS and stem in m.stem]
            mask = matches[0] if matches else None
        if mask is not None:
            pairs.append((img, mask))
    return pairs


# ---------------------------------------------------------------------------
# MinneApple
# ---------------------------------------------------------------------------
def _coco_split_jsons(src: Path) -> List[Tuple[str, Path]]:
    """Найти COCO-сплиты вида <split>/{*.json + images/} (MinneApple с HF)."""
    found: List[Tuple[str, Path]] = []
    for d in [src, *sorted(p for p in src.iterdir() if p.is_dir())]:
        if not (d / "images").is_dir():
            continue
        for j in sorted(d.glob("*.json")):
            try:
                data = json.loads(j.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            if isinstance(data, dict) and {"images", "annotations"} <= set(data):
                name = d.name.lower() if d != src else j.stem.lower()
                found.append((name, j))
                break
    return found


def _carve_val(out_path: Path, ratio: float = 0.1) -> Tuple[int, int]:
    """Гарантировать непустой val: выдержать долю из train (и метки вместе)."""
    val_img = out_path / "images" / "val"
    val_lbl = out_path / "labels" / "val"
    train_img = out_path / "images" / "train"
    train_lbl = out_path / "labels" / "train"
    val_img.mkdir(parents=True, exist_ok=True)
    val_lbl.mkdir(parents=True, exist_ok=True)
    files = sorted(train_img.iterdir()) if train_img.exists() else []
    move_n = max(1, int(len(files) * ratio)) if files else 0
    for f in files[:move_n]:
        f.rename(val_img / f.name)
        lbl = train_lbl / f"{f.stem}.txt"
        if lbl.exists():
            lbl.rename(val_lbl / lbl.name)
    return len(files) - move_n, move_n


def _convert_minneapple_coco(out_path: Path, splits: List[Tuple[str, Path]],
                             val_ratio: float, seed: int) -> dict:
    """MinneApple (COCO JSON: bbox [x,y,w,h]) -> YOLO-датасет."""
    rng = random.Random(seed)
    has_eval = any(name in ("val", "valid", "test") for name, _ in splits)
    n_train = n_val = n_boxes = 0
    for split_name, json_path in splits:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        images = {int(im["id"]): im for im in data.get("images", [])}
        anns: Dict[int, list] = {}
        for a in data.get("annotations", []):
            if a.get("iscrowd"):
                continue
            anns.setdefault(int(a["image_id"]), []).append(a)
        img_dir = json_path.parent / "images"
        is_eval = split_name in ("val", "valid", "test")
        for im in images.values():
            img_path = img_dir / im["file_name"]
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            h, w = image.shape[:2]
            boxes: List[Tuple[float, float, float, float]] = []
            for a in anns.get(int(im["id"]), []):
                bbox = a.get("bbox") or []
                if len(bbox) != 4:
                    continue
                x, y, bw, bh = (float(v) for v in bbox)
                boxes.append((x, y, x + bw, y + bh))
            target = "val" if is_eval else "train"
            if target == "train" and not has_eval and rng.random() < val_ratio:
                target = "val"
            dst_img = out_path / "images" / target
            dst_lbl = out_path / "labels" / target
            dst_img.mkdir(parents=True, exist_ok=True)
            dst_lbl.mkdir(parents=True, exist_ok=True)
            name = im["file_name"]
            cv2.imwrite(str(dst_img / name), image)
            _write_yolo_label(dst_lbl / f"{Path(name).stem}.txt", boxes, w, h)
            n_boxes += len(boxes)
            if target == "train":
                n_train += 1
            else:
                n_val += 1

    if n_train == 0:
        raise ValueError("не удалось сконвертировать ни одного изображения MinneApple (COCO)")
    if n_boxes == 0:
        raise ValueError("COCO-аннотации не дали ни одного бокса")
    if n_val == 0:
        n_train, n_val = _carve_val(out_path)
        logger.warning("val-сплит отсутствовал: выдержан из train (%d изображений)", n_val)
    yaml_path = _write_data_yaml(out_path, n_train, n_val)
    return {
        "dataset": "minneapple", "format": "coco",
        "train": n_train, "val": n_val, "boxes": n_boxes,
        "data_yaml": str(yaml_path),
    }


def convert_minneapple(src: str, out: str, val_ratio: float = 0.15, seed: int = 42) -> dict:
    """MinneApple -> YOLO-датасет.

    Поддерживаются две раскладки:
      * COCO-JSON (HuggingFace: <split>/{images/, *.json} с bbox);
      * классическая (imgs + masks сегментации).
    """
    src_path = Path(src)
    out_path = Path(out)
    if not src_path.exists():
        raise FileNotFoundError(f"каталог MinneApple не найден: {src}")

    coco = _coco_split_jsons(src_path)
    if coco:
        return _convert_minneapple_coco(out_path, coco, val_ratio, seed)

    images_root = _find_dirs(src_path, ["imgs", "images"])
    masks_root = _find_dirs(src_path, ["masks", "mask"])
    if images_root is None or masks_root is None:
        raise ValueError(
            f"не найдены каталоги imgs/masks внутри {src_path}; "
            f"структура: {[p.name for p in src_path.iterdir() if p.is_dir()][:10]}"
        )

    # MinneApple: dataset/{train,val}/{imgs,masks}
    split_dirs: Dict[str, Tuple[Path, Path]] = {}
    for split in ("train", "val", "test"):
        img_dir = src_path / split / "imgs"
        mask_dir = src_path / split / "masks"
        if not img_dir.exists():
            img_dir = src_path / split / "images"
        if img_dir.exists() and mask_dir.exists():
            split_dirs[split] = (img_dir, mask_dir)

    if not split_dirs:
        split_dirs["train"] = (images_root, masks_root)

    rng = random.Random(seed)
    n_train = n_val = 0
    for split, (img_dir, mask_dir) in split_dirs.items():
        pairs = _pairs_from_dirs(img_dir, mask_dir)
        for img_path, mask_path in pairs:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            h, w = image.shape[:2]
            boxes = mask_to_boxes(mask)
            target_split = "val" if split in ("val", "test") else "train"
            # случайная выдержка из train в val
            if target_split == "train" and "val" not in split_dirs and rng.random() < val_ratio:
                target_split = "val"
            dst_img = out_path / "images" / target_split
            dst_lbl = out_path / "labels" / target_split
            dst_img.mkdir(parents=True, exist_ok=True)
            dst_lbl.mkdir(parents=True, exist_ok=True)
            suffix = img_path.suffix.lower()
            name = f"{split}_{img_path.stem}{suffix}"
            cv2.imwrite(str(dst_img / name), image)
            _write_yolo_label(dst_lbl / f"{name.rsplit('.', 1)[0]}.txt", boxes, w, h)
            if target_split == "train":
                n_train += 1
            else:
                n_val += 1

    if n_train == 0:
        raise ValueError("не удалось сконвертировать ни одного изображения MinneApple")
    if n_val == 0:
        # гарантируем непустой val
        val_img = out_path / "images" / "val"
        val_lbl = out_path / "labels" / "val"
        train_img = out_path / "images" / "train"
        train_lbl = out_path / "labels" / "train"
        files = sorted(train_img.iterdir())
        move_n = max(1, int(len(files) * 0.1))
        for f in files[:move_n]:
            f.rename(val_img / f.name)
            lbl = train_lbl / f"{f.stem}.txt"
            if lbl.exists():
                lbl.rename(val_lbl / lbl.name)
        n_val, n_train = move_n, n_train - move_n

    yaml_path = _write_data_yaml(out_path, n_train, n_val)
    return {"dataset": "minneapple", "train": n_train, "val": n_val, "data_yaml": str(yaml_path)}


# ---------------------------------------------------------------------------
# ACFR
# ---------------------------------------------------------------------------
def convert_acfr(src: str, out: str, val_ratio: float = 0.15, seed: int = 42) -> dict:
    """ACFR MultiFruit (apples/) -> YOLO-датасет."""
    src_path = Path(src)
    out_path = Path(out)
    if not src_path.exists():
        raise FileNotFoundError(f"каталог ACFR не найден: {src}")

    # поддерживаем и корень архива (…/acfr-fruit-dataset/apples), и сам каталог apples
    if src_path.name.lower() != "apples":
        candidate = src_path / "acfr-fruit-dataset" / "apples"
        if candidate.exists():
            src_path = candidate
        else:
            found = [p for p in src_path.rglob("*") if p.is_dir() and p.name.lower() == "apples"]
            if found:
                src_path = found[0]

    images_dir = _find_dirs(src_path, ["images", "imgs"])
    # приоритет: бинарные маски (segmentations) важнее CSV-аннотаций
    seg_dir = None
    for name in ("segmentations", "masks", "mask", "annotations"):
        seg_dir = _find_dirs(src_path, [name])
        if seg_dir is not None:
            break
    if images_dir is None or seg_dir is None:
        raise ValueError(
            f"не найдены images/segmentations внутри {src_path}; "
            f"структура: {[p.name for p in src_path.iterdir() if p.is_dir()][:10]}"
        )

    # сплиты из sets/*.txt
    splits: Dict[str, set] = {}
    sets_dir = src_path / "sets"
    if sets_dir.exists():
        for txt in sets_dir.glob("*.txt"):
            lines = [ln.strip() for ln in txt.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
            keys = {_split_key(ln) for ln in lines}
            name = txt.stem.lower()
            if name == "train":          # train_val не берём (это объединение)
                splits["train"] = splits.get("train", set()) | keys
            elif name in ("val", "valid", "test"):
                splits["val"] = splits.get("val", set()) | keys

    rng = random.Random(seed)
    n_train = n_val = 0
    n_boxes = 0
    ann_dir = src_path / "annotations"
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    for img_path in images:
        stem = img_path.stem
        boxes = _parse_acfr_annotations(ann_dir, stem) if ann_dir.exists() else None
        mask_path = None
        if boxes is None:
            for cand_stem in (stem, f"{stem}_L"):  # ACFR: маски именуются <stem>_L.png
                for ext in (".png", ".jpg", ".tif", ".tiff"):
                    cand = seg_dir / f"{cand_stem}{ext}"
                    if cand.exists():
                        mask_path = cand
                        break
                if mask_path is not None:
                    break
            if mask_path is None:
                # «мягкое» совпадение по подстроке (если каталог смешанный)
                matches = [m for m in seg_dir.iterdir()
                           if m.suffix.lower() in IMG_EXTS and stem in m.stem]
                mask_path = matches[0] if matches else None
            if mask_path is None:
                continue
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue
            boxes = mask_to_boxes(mask)

        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]

        if "train" in splits or "val" in splits:
            target = "val" if stem in splits.get("val", set()) else "train"
        else:
            target = "val" if rng.random() < val_ratio else "train"

        dst_img = out_path / "images" / target
        dst_lbl = out_path / "labels" / target
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        name = img_path.name
        cv2.imwrite(str(dst_img / name), image)
        _write_yolo_label(dst_lbl / f"{img_path.stem}.txt", boxes, w, h)
        n_boxes += len(boxes)
        if target == "train":
            n_train += 1
        else:
            n_val += 1

    if n_train == 0:
        raise ValueError("не удалось сконвертировать ни одного изображения ACFR")
    if n_boxes == 0:
        raise ValueError(
            "не найдено ни одного бокса: проверьте annotations/*.csv или segmentations/*_L.png"
        )
    if n_val == 0:
        # сплит не сработал (нет sets/ или не совпали имена) — выдерживаем val из train
        val_img = out_path / "images" / "val"
        val_lbl = out_path / "labels" / "val"
        train_img = out_path / "images" / "train"
        train_lbl = out_path / "labels" / "train"
        val_img.mkdir(parents=True, exist_ok=True)
        val_lbl.mkdir(parents=True, exist_ok=True)
        files = sorted(train_img.iterdir())
        move_n = max(1, int(len(files) * val_ratio))
        for f in files[:move_n]:
            f.rename(val_img / f.name)
            lbl = train_lbl / f"{f.stem}.txt"
            if lbl.exists():
                lbl.rename(val_lbl / lbl.name)
        n_val, n_train = move_n, n_train - move_n
        logger.warning("sets/ не сработал: val выдержан из train (%d изображений)", n_val)
    yaml_path = _write_data_yaml(out_path, n_train, n_val)
    return {"dataset": "acfr", "train": n_train, "val": n_val, "data_yaml": str(yaml_path)}


CONVERTERS = {"minneapple": convert_minneapple, "acfr": convert_acfr}


def convert_dataset(name: str, src: str, out: str, **kwargs) -> dict:
    name = name.lower()
    if name not in CONVERTERS:
        raise ValueError(f"нет конвертера для {name!r}; доступны: {', '.join(CONVERTERS)}")
    result = CONVERTERS[name](src, out, **kwargs)
    logger.info("конвертация завершена: %s", json.dumps(result, ensure_ascii=False))
    return result
