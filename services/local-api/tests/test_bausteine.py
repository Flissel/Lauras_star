"""Reel-Bausteine: Standbilder mit Bewegung und Titelkarten (Video-Claw, 17.10.2026)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from laura.bausteine import filter as bf
from laura.config import Settings
from laura.db import repos
from laura.ingest.ffmpeg import probe as ffprobe
from laura.ingest.ffmpeg import run_ffmpeg
from laura.ingest.probe import parse_probe
from laura.main import create_app


# --- reine Teile -------------------------------------------------------------

@pytest.mark.parametrize("bewegung", bf.BEWEGUNGEN)
def test_standbild_graph(bewegung: str) -> None:
    g = bf.standbild_graph(bewegung, 90)
    assert "zoompan=" in g and ":d=90:" in g and "s=1080x1920" in g and g.endswith("[v]")
    assert "boxblur" in g and "force_original_aspect_ratio=decrease" in g   # Blur-Fill


def test_standbild_graph_ungueltig() -> None:
    with pytest.raises(ValueError):
        bf.standbild_graph("drehen", 90)
    with pytest.raises(ValueError):
        bf.standbild_graph("ruhig", 0)


@pytest.mark.parametrize("roh", ["red", "#12345", "#1234567", "#12345g", "#123456;x", None])
def test_farbe_nur_hex(roh: Any) -> None:
    with pytest.raises(ValueError):
        bf.farbe_ffmpeg(roh)


def test_farbe_ok() -> None:
    assert bf.farbe_ffmpeg("#A1b2C3") == "0xa1b2c3"


def test_umbrechen() -> None:
    assert bf.umbrechen("Drei Fehler") == (["Drei Fehler"], 96)
    zeilen, gr = bf.umbrechen("Was Unternehmen bei der bAV wirklich falsch machen")
    assert gr in (80, 68) and all(len(z) <= 23 for z in zeilen) and len(zeilen) <= 4
    wort = "Ueberversicherungsvermeidungsstrategie"
    zeilen, gr = bf.umbrechen(wort)
    assert "".join(zeilen) == wort and gr == 80 and all(len(z) <= 19 for z in zeilen)
    with pytest.raises(ValueError):
        bf.umbrechen("   ")
    with pytest.raises(ValueError):
        bf.umbrechen(" ".join(["Wort"] * 30))


def test_titelkarte_graph_mit_logo() -> None:
    g = bf.titelkarte_graph(["z0.txt", "z1.txt"], schriftgroesse=80, font="f.ttf",
                            textfarbe="#ffffff", unterzeile_datei="u.txt", mit_logo=True)
    assert g.count("drawtext=") == 3 and "textfile=z1.txt" in g and "[1:v]scale=" in g
    assert "fade=t=in" in g and "0xffffff" in g


# --- Bilderkennung -----------------------------------------------------------

def test_probe_png_ist_bild() -> None:
    pr = parse_probe({"format": {"format_name": "png_pipe"},
                      "streams": [{"codec_type": "video", "codec_name": "png", "width": 800,
                                   "height": 600, "r_frame_rate": "25/1"}]})
    assert pr.type == "image" and pr.width == 800 and pr.duration_frames is None


def test_probe_mjpeg_video_bleibt_video() -> None:
    pr = parse_probe({"format": {"format_name": "avi", "duration": "10"},
                      "streams": [{"codec_type": "video", "codec_name": "mjpeg", "width": 640,
                                   "height": 480, "r_frame_rate": "25/1", "avg_frame_rate": "25/1",
                                   "nb_frames": "250"}]})
    assert pr.type == "video" and pr.duration_frames == 250


# --- echte Laeufe ------------------------------------------------------------

def _client(tmp_path: Path) -> tuple[TestClient, Any]:
    app = create_app(Settings(workspace_root=tmp_path, token=None, start_runner=False))
    return TestClient(app), app


def _bild_asset(app: Any, pid: str, tmp_path: Path) -> str:
    png = tmp_path / "bild.png"
    run_ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=1600x900", "-frames:v", "1", str(png)])
    pr = parse_probe(ffprobe(png))
    a = repos.create_asset(app.state.db, project_id=pid, type=pr.type, display_name="bild.png",
                           source_path=str(png))
    return str(a["id"])


def _projekt(client: TestClient) -> str:
    return str(client.post("/projects", json={"name": "p", "sequence_rate_num": 30,
                                              "sequence_rate_den": 1}).json()["id"])


def _job_ergebnis(app: Any, job_id: str) -> dict[str, Any]:
    for _ in range(10):
        if not app.state.runner.run_once():
            break
    job = repos.get_job(app.state.db, job_id)
    assert job is not None and job["status"] == "succeeded", job
    return dict(json.loads(job["result_json"]))


def test_standbild_echt(tmp_path: Path) -> None:
    client, app = _client(tmp_path)
    pid = _projekt(client)
    bild = _bild_asset(app, pid, tmp_path)
    r = client.post(f"/projects/{pid}/stills", json={"asset_id": bild, "dauer_s": 2,
                                                     "bewegung": "zoom_rein"})
    assert r.status_code == 202, r.text
    erg = _job_ergebnis(app, r.json()["job_id"])
    a = repos.get_asset(app.state.db, erg["asset_id"])
    assert a["type"] == "video" and a["width"] == 1080 and a["height"] == 1920
    assert a["duration_frames"] == 60 and a["audio_sample_rate"] == 48000
    assert not a["synthetic"]


def test_titelkarte_echt_mit_sonderzeichen_und_logo(tmp_path: Path) -> None:
    client, app = _client(tmp_path)
    pid = _projekt(client)
    logo = _bild_asset(app, pid, tmp_path)
    r = client.post(f"/projects/{pid}/titelkarte", json={
        "text": "Drei Fehler: was 'wirklich' zaehlt, 100 %", "unterzeile": "bAV; kurz erklaert",
        "hintergrund": "#0f2422", "textfarbe": "#e9fbf6", "logo_asset_id": logo, "dauer_s": 2})
    assert r.status_code == 202, r.text
    erg = _job_ergebnis(app, r.json()["job_id"])
    a = repos.get_asset(app.state.db, erg["asset_id"])
    assert a["type"] == "video" and a["height"] == 1920 and a["duration_frames"] == 60
    assert not list((Path(a["source_path"]).parent).glob("tmp-*"))     # Arbeitsordner weg


def test_api_kein_bild_422(tmp_path: Path) -> None:
    client, app = _client(tmp_path)
    pid = _projekt(client)
    v = repos.create_asset(app.state.db, project_id=pid, type="video", display_name="v.mp4",
                           source_path=str(tmp_path / "v.mp4"))
    assert client.post(f"/projects/{pid}/stills", json={"asset_id": v["id"]}).status_code == 422


def test_api_unbekanntes_projekt_404(tmp_path: Path) -> None:
    client, _app = _client(tmp_path)
    assert client.post("/projects/nope/titelkarte",
                       json={"text": "x", "hintergrund": "#000000"}).status_code == 404


@pytest.mark.parametrize("koerper", [
    {"text": "x", "hintergrund": "schwarz"},
    {"text": "x", "hintergrund": "#000000", "textfarbe": "#fff"},
    {"text": "", "hintergrund": "#000000"},
    {"text": "x", "hintergrund": "#000000", "dauer_s": 20},
])
def test_api_titelkarte_formen_422(tmp_path: Path, koerper: dict[str, Any]) -> None:
    client, _app = _client(tmp_path)
    pid = _projekt(client)
    assert client.post(f"/projects/{pid}/titelkarte", json=koerper).status_code == 422
