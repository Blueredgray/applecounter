"""Загрузка общедоступных датасетов для обучения и валидации.

  * MinneApple        — https://conservancy.umn.edu/handle/11299/206575 (страница DSpace)
  * ACFR MultiFruit   — https://data.acfr.usyd.edu.au/ag/treecrops/2016-multifruit
                        (прямой архив acfr-multifruit-2016.zip, ~2.2 ГБ)
  * VideoFruitCounting — https://github.com/imatge-upc/VideoFruitCounting
                        (видео/веса/SFRAME выложены на gofile, ссылки в README репозитория)

Скачивание выполняется с возобновлением (Range), учитывается SHA-256;
если сайт отвечает 403 (антибот), печатаются инструкции для ручной загрузки.
"""

from __future__ import annotations

import hashlib
import json
import re
import ssl
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from ..utils.logging import get_logger

logger = get_logger()
CTX = ssl.create_default_context()
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

ACFR_URL = "https://data.acfr.usyd.edu.au/ag/treecrops/2016-multifruit/acfr-multifruit-2016.zip"
MINNEAPPLE_HANDLE = "https://conservancy.umn.edu/handle/11299/206575"
VIDEOFRUIT_REPO = "https://github.com/imatge-upc/VideoFruitCounting"
VIDEOFRUIT_FILES = {
    "videos": "https://gofile.me/73Gps/3VE9yy9aX",
    "yolo_weights": "https://gofile.me/73Gps/clQj1gJhv",
    "sframe": "https://gofile.me/73Gps/DECp9orob",
}


@dataclass
class DatasetSpec:
    name: str
    title: str
    urls: List[str] = field(default_factory=list)
    page: Optional[str] = None
    notes: str = ""


DATASETS = {
    "minneapple": DatasetSpec(
        name="minneapple",
        title="MinneApple (Univ. of Minnesota) — сегментация/детекция яблок",
        urls=[],
        page=MINNEAPPLE_HANDLE,
        notes=(
            "DSpace защищён от автоматических загрузок (HTTP 403). Откройте страницу "
            "в браузере и скачайте архив датасета, затем:\n"
            "  python -m applecounter download --dataset minneapple --archive путь/к/архиву.zip"
        ),
    ),
    "acfr": DatasetSpec(
        name="acfr",
        title="ACFR Orchard Fruit Dataset (MultiFruit 2016) — яблоки/манго/миндаль",
        urls=[ACFR_URL],
        page="https://data.acfr.usyd.edu.au/ag/treecrops/2016-multifruit",
        notes="Архив ~2.2 ГБ, загрузка с возобновлением; распаковываются только яблоки (apples/).",
    ),
    "videofruit": DatasetSpec(
        name="videofruit",
        title="VideoFruitCounting (GPI/UPC, IRTA) — видео пролёта вдоль ряда + SFRAME",
        urls=[],
        page=VIDEOFRUIT_REPO,
        notes=(
            "Файлы выложены на gofile и требуют ручной загрузки по ссылкам из README:\n"
            + "\n".join(f"  - {k}: {v}" for k, v in VIDEOFRUIT_FILES.items())
            + "\nПосле скачивания положите файлы в data/raw/videofruit/ или укажите --url."
        ),
    ),
}


def _requests_get(url: str, headers: Optional[dict] = None, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as resp:
        return resp.read()


def download_file(url: str, dest: Path, resume: bool = True, chunk: int = 1 << 20) -> Path:
    """Скачать файл с поддержкой докачки; печатать прогресс."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    done = part.stat().st_size if part.exists() and resume else 0
    headers = {"User-Agent": USER_AGENT}
    if done:
        headers["Range"] = f"bytes={done}-"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120, context=CTX) as resp:
            total = resp.headers.get("Content-Length")
            total = int(total) + done if total else None
            mode = "ab" if done and resp.status == 206 else "wb"
            if mode == "wb":
                done = 0
            with part.open(mode) as fh:
                while True:
                    chunk_data = resp.read(chunk)
                    if not chunk_data:
                        break
                    fh.write(chunk_data)
                    done += len(chunk_data)
                    pct = f"{100.0 * done / total:.1f}%" if total else "?"
                    sys.stdout.write(f"\r  {dest.name}: {done / 1e6:.1f} МБ ({pct})")
                    sys.stdout.flush()
        sys.stdout.write("\n")
    except urllib.error.HTTPError as exc:
        if exc.code == 416:  # диапазон уже скачан целиком
            pass
        else:
            raise
    if part.exists():
        part.replace(dest)
    return dest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_minneapple_bitstream(page_url: str) -> Optional[str]:
    """Попытка найти прямую ссылку DSpace bitstream на странице handle."""
    try:
        html = _requests_get(page_url).decode("utf-8", "ignore")
    except Exception as exc:  # 403/таймаут — ожидаемо
        logger.warning("страница MinneApple недоступна автоматически: %s", str(exc)[:120])
        return None
    matches = re.findall(r'href="([^"]*bitstreams[^"]*/download[^"]*)"', html)
    if not matches:
        matches = re.findall(r'href="([^"]*bitstream[^"]*\.zip[^"]*)"', html)
    if not matches:
        return None
    link = matches[0]
    if link.startswith("/"):
        link = "https://conservancy.umn.edu" + link
    return link


def unzip(path: Path, out_dir: Path, members_prefix: Optional[str] = None) -> Path:
    """Распаковать zip (опционально только часть архива по префиксу)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        if members_prefix:
            names = [n for n in names if n.startswith(members_prefix)]
        if not names:
            raise ValueError(f"в архиве нет записей с префиксом {members_prefix!r}")
        total = sum(i.file_size for i in zf.infolist() if not i.is_dir())
        logger.info("распаковка %d файлов (%.1f МБ) -> %s", len(names), total / 1e6, out_dir)
        for name in names:
            zf.extract(name, out_dir)
    return out_dir


def _write_instructions(out: Path, spec: DatasetSpec, result: dict) -> None:
    """Сохранить INSTRUCTIONS.txt при ручном режиме, чтобы каталог не был пустым."""
    text = [
        f"{spec.title}",
        f"страница: {spec.page}" if spec.page else "",
        "",
        (result.get("message") or ""),
        "",
        spec.notes,
        "",
        "После ручной загрузки выполните:",
        f"  python -m applecounter download --dataset {spec.name} --archive путь/к/архиву.zip",
        "",
        "Итог этой попытки:",
        json.dumps(result, ensure_ascii=False, indent=2),
    ]
    (out / "INSTRUCTIONS.txt").write_text("\n".join(x for x in text if x != ""), encoding="utf-8")
    (out / "download.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def download_dataset(name: str, out_dir: str, url: Optional[str] = None,
                     archive: Optional[str] = None, keep_archive: bool = False) -> dict:
    """Скачать/подготовить датасет. Возвращает словарь с путями."""
    name = name.lower()
    if name not in DATASETS:
        raise ValueError(f"неизвестный датасет {name!r}; доступны: {', '.join(DATASETS)}")
    spec = DATASETS[name]
    out = Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    result = {"name": name, "dir": str(out), "notes": spec.notes, "page": spec.page}

    if archive:
        archive_path = Path(archive)
        if not archive_path.exists():
            raise FileNotFoundError(f"архив не найден: {archive_path}")
        logger.info("используется локальный архив %s", archive_path)
        url = url or (spec.urls[0] if spec.urls else None)
    elif url:
        archive_path = out / Path(url).name
        if not archive_path.exists():
            logger.info("скачивание %s", url)
            download_file(url, archive_path)
    elif spec.urls:
        archive_path = out / Path(spec.urls[0]).name
        if not archive_path.exists():
            logger.info("скачивание %s", spec.urls[0])
            download_file(spec.urls[0], archive_path)
    else:
        # нет прямой ссылки — пробуем страницу, иначе инструкции
        if name == "minneapple":
            link = _find_minneapple_bitstream(spec.page or MINNEAPPLE_HANDLE)
            if link:
                archive_path = out / Path(link).name.split("?")[0]
                if not archive_path.exists():
                    logger.info("скачивание %s", link)
                    download_file(link, archive_path)
            else:
                result["downloaded"] = False
                result["message"] = "Прямая ссылка не найдена — скачайте вручную и повторите с --archive."
                _write_instructions(out, spec, result)
                return result
        else:
            result["downloaded"] = False
            result["message"] = "Прямой ссылки нет — скачайте вручную (см. notes) и повторите с --archive/--url."
            _write_instructions(out, spec, result)
            return result

    result["archive"] = str(archive_path)
    result["sha256"] = sha256(archive_path)

    if name == "acfr":
        unzip(archive_path, out, members_prefix="acfr-fruit-dataset/apples")
        result["dataset_dir"] = str(out / "acfr-fruit-dataset" / "apples")
    else:
        unzip(archive_path, out)
        result["dataset_dir"] = str(out)

    if not keep_archive and name == "acfr":
        logger.info("архив %.1f ГБ оставлен в %s (удалите вручную при нехватке места)",
                    archive_path.stat().st_size / 1e9, archive_path)
    result["downloaded"] = True
    (out / "download.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def list_datasets() -> str:
    lines = []
    for spec in DATASETS.values():
        lines.append(f"{spec.name:12s} — {spec.title}")
        if spec.page:
            lines.append(f"{'':12s}   страница: {spec.page}")
        for url in spec.urls:
            lines.append(f"{'':12s}   прямая ссылка: {url}")
        if spec.notes:
            lines.append("  " + spec.notes.replace("\n", "\n  "))
        lines.append("")
    return "\n".join(lines)
