import zipfile
from pathlib import Path

from applecounter.data.download import DATASETS, download_file, sha256, unzip


def test_datasets_registry():
    assert set(DATASETS) == {"minneapple", "acfr", "videofruit"}
    acfr = DATASETS["acfr"]
    assert acfr.urls and "acfr-multifruit-2016.zip" in acfr.urls[0]
    assert "conservancy.umn.edu" in DATASETS["minneapple"].page
    assert "github.com" in DATASETS["videofruit"].page


def test_sha256(tmp_path):
    f = tmp_path / "blob.bin"
    f.write_bytes(b"hello world")
    digest = sha256(f)
    assert digest == ("b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9")


def test_download_file_from_file_url(tmp_path):
    src = tmp_path / "src.txt"
    src.write_bytes(b"x" * 5000)
    dest = tmp_path / "dst.txt"
    out = download_file(src.resolve().as_uri(), dest, resume=False)
    assert out.read_bytes() == src.read_bytes()


def test_unzip_selective(tmp_path):
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("root/apples/images/1.png", b"img")
        zf.writestr("root/apples/masks/1.png", b"mask")
        zf.writestr("root/mango/images/9.png", b"other")
    out = unzip(archive, tmp_path / "out", members_prefix="root/apples")
    assert (out / "root" / "apples" / "images" / "1.png").exists()
    assert not (out / "root" / "mango").exists()


def test_unzip_missing_prefix(tmp_path):
    import pytest

    archive = tmp_path / "b.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("x/y.txt", b"1")
    with pytest.raises(ValueError):
        unzip(archive, tmp_path / "out2", members_prefix="нет/")
