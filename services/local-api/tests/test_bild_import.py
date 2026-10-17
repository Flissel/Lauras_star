"""Bilder per Direktlink importieren (Video-Claw, 17.10.2026)."""
from __future__ import annotations

from pathlib import Path

import pytest

from laura.ingest.ffmpeg import probe, run_ffmpeg
from laura.ingest.integrity import verify_decode
from laura.ingest.probe import parse_probe
from laura.ingest.ytdlp import needs_ytdlp


@pytest.mark.parametrize("url", ["https://m.example/api/medien/datei/bild.png?t=x",
                                 "https://m.example/a/logo.JPG", "https://m.example/b.webp"])
def test_bild_url_ist_direktlink(url: str) -> None:
    assert needs_ytdlp(url) is False


@pytest.mark.parametrize("endung", ["png", "jpg"])
def test_bild_besteht_pruefung_und_ist_image(tmp_path: Path, endung: str) -> None:
    bild = tmp_path / f"b.{endung}"
    run_ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=640x360", "-frames:v", "1", str(bild)])
    assert verify_decode(bild, full_scan=True).ok
    assert parse_probe(probe(bild)).type == "image"
