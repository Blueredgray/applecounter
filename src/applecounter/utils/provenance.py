"""Run provenance and content hashes used to prevent silent result mixing."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_source(path: str | Path) -> str:
    """Hash a video file or every image in an image-sequence directory."""
    source = Path(path)
    if source.is_file():
        return sha256_file(source)
    if not source.is_dir():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    files = sorted(p for p in source.rglob("*") if p.is_file())
    for item in files:
        digest.update(item.relative_to(source).as_posix().encode("utf-8"))
        digest.update(sha256_file(item).encode("ascii"))
    return digest.hexdigest()


def package_versions() -> dict:
    names = ("applecounter", "numpy", "opencv-python", "scipy", "pandas", "ultralytics",
             "torch", "boxmot", "deep-sort-realtime", "motmetrics")
    result = {"python": sys.version.split()[0]}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def git_commit(root: str | Path) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(root), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
        )
        return proc.stdout.strip() if proc.returncode == 0 else None
    except OSError:
        return None


def detection_cache_metadata(cfg, source: str | Path) -> dict:
    """Configuration fields that must match before a cache can be reused."""
    weights = cfg.get("detector.yolo.weights")
    weights_hash = sha256_file(weights) if weights and Path(str(weights)).is_file() else None
    return {
        "source": str(Path(source).resolve()),
        "source_sha256": sha256_source(source),
        "detector": cfg.get("detector", {}),
        "weights_sha256": weights_hash,
        "video": {
            "frame_stride": int(cfg.get("video.frame_stride", 1)),
            "resize": cfg.get("video.resize"),
            "rotate90": int(cfg.get("video.rotate90", 0)),
            "max_frames": cfg.get("video.max_frames"),
        },
    }


def write_run_manifest(path: str | Path, cfg, source: str | Path, status: str = "complete") -> Path:
    root = Path(__file__).resolve().parents[3]
    data = {
        "status": status,
        "source": str(Path(source).resolve()),
        "source_sha256": sha256_source(source),
        "config": cfg.data,
        "packages": package_versions(),
        "git_commit": git_commit(root),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    return path
