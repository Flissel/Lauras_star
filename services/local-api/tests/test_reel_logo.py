"""Logo-Einblendung im Reel (Video-Claw, 17.10.2026)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from laura.config import Settings
from laura.db import repos
from laura.ingest.ffmpeg import probe, run_ffmpeg
from laura.main import create_app
from laura.render import logo

from .test_reel_groesse import _job


@pytest.mark.parametrize("vb, lb, lh, p, erwartet", [
    (1080, 1600, 900, 14, (152, 86)),
    (1080, 500, 500, 10, (108, 108)),
    (1080, 300, 900, 8, (86, 258)),
])
def test_logo_masse(vb: int, lb: int, lh: int, p: int, erwartet: tuple[int, int]) -> None:
    assert logo.logo_masse(vb, lb, lh, p) == erwartet


@pytest.mark.parametrize("p", [7, 26])
def test_logo_masse_grenzen(p: int) -> None:
    with pytest.raises(ValueError):
        logo.logo_masse(1080, 100, 100, p)


def test_logo_graph() -> None:
    assert logo.logo_graph("oben_rechts", 152, 86) == (
        "[1:v]scale=152:86[l];[0:v][l]overlay=W-w-48:140,format=yuv420p[v]")
    with pytest.raises(ValueError):
        logo.logo_graph("unten", 10, 10)


def test_einblenden_echt(tmp_path: Path) -> None:
    video, bild = tmp_path / "v.mp4", tmp_path / "l.png"
    run_ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=1080x1920:rate=30", "-f", "lavfi", "-i", "sine",
                "-t", "1", "-c:v", "libx264", "-c:a", "aac", str(video)])
    run_ffmpeg(["-f", "lavfi", "-i", "color=c=red:s=400x200", "-frames:v", "1", str(bild)])
    logo.einblenden(video, bild, position="oben_links", breite_prozent=20)
    typen = {s["codec_type"]: s for s in probe(video)["streams"]}
    assert typen["video"]["width"] == 1080 and "audio" in typen
    assert not list(tmp_path.glob("*.logo.mp4"))


def test_job_logo_vor_groesse(tmp_path: Path, monkeypatch: Any) -> None:
    from laura.render import groesse
    reihenfolge: list[str] = []
    monkeypatch.setattr(logo, "einblenden",
                        lambda d, l, **k: reihenfolge.append(f"logo:{k['position']}:{k['breite_prozent']}"))
    monkeypatch.setattr(groesse, "einhalten",
                        lambda d, **k: reihenfolge.append("groesse") or 14_000_000)
    from laura.db.database import SqliteDatabase  # noqa: F401  (Fixture baut die DB selbst)
    orig = repos.get_asset
    monkeypatch.setattr(repos, "get_asset",
                        lambda db, aid: {"source_path": "logo.png"} if aid == "LOGO" else orig(db, aid))
    _db, exp = _job(tmp_path, monkeypatch, {"vertical": True, "max_bytes": 15_000_000,
                                            "logo_asset_id": "LOGO", "logo_position": "oben_mitte",
                                            "logo_breite_prozent": 12})
    assert reihenfolge == ["logo:oben_mitte:12", "groesse"] and exp["status"] == "ready"


def _client(tmp_path: Path) -> tuple[TestClient, Any]:
    app = create_app(Settings(workspace_root=tmp_path, token=None, start_runner=False))
    return TestClient(app), app.state.db


def test_api_logo_muss_bild_sein(tmp_path: Path) -> None:
    client, db = _client(tmp_path)
    pid = client.post("/projects", json={"name": "p", "sequence_rate_num": 30,
                                         "sequence_rate_den": 1}).json()["id"]
    tl = repos.create_timeline(db, project_id=pid, name="cut", kind="rough_cut")
    v = repos.create_asset(db, project_id=pid, type="video", display_name="v", source_path="v.mp4")
    b = repos.create_asset(db, project_id=pid, type="image", display_name="b", source_path="b.png")
    assert client.post(f"/timelines/{tl['id']}/render-reel",
                       json={"logo_asset_id": v["id"]}).status_code == 422
    r = client.post(f"/timelines/{tl['id']}/render-reel", json={"logo_asset_id": b["id"]})
    assert r.status_code == 202
    opts = repos.get_export(db, r.json()["export_id"])["options"]
    assert opts["logo_asset_id"] == b["id"] and opts["logo_position"] == "oben_rechts"
