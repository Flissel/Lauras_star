"""Groessengrenze fuer Reels (Video-Claw, 17.10.2026).

sales-claw nimmt hoechstens 15 MB je Video (LinkedIn/WhatsApp). ``max_bytes`` am
Reel-Render: ist die fertige Datei groesser, wird sie mit einer aus der Dauer
berechneten Bitrate neu kodiert; ein zweiter Versuch mit 80 %, danach ein Fehler
mit Grund. Ohne ``max_bytes`` aendert sich nichts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from laura.api.models import ReelRenderRequest
from laura.render import groesse


@pytest.mark.parametrize("max_bytes, dauer_s, erwartet", [
    (15_000_000, 30.0, 3_552),   # 15 MB*8*0.92/30 s = 3680 kbit/s - 128 Ton
    (15_000_000, 60.0, 1_712),
    (15_000_000, 600.0, 300),    # Untergrenze: darunter wird es Matsch
])
def test_ziel_bitrate(max_bytes: int, dauer_s: float, erwartet: int) -> None:
    assert groesse.ziel_bitrate_kbps(max_bytes, dauer_s) == erwartet


def test_ziel_bitrate_ungueltig() -> None:
    with pytest.raises(ValueError):
        groesse.ziel_bitrate_kbps(15_000_000, 0)


def _datei(tmp_path: Path, groesse_bytes: int) -> Path:
    p = tmp_path / "e.mp4"
    p.write_bytes(b"\0" * groesse_bytes)
    return p


def test_klein_genug_bleibt_unangetastet(tmp_path: Path, monkeypatch: Any) -> None:
    aufrufe: list[list[str]] = []
    monkeypatch.setattr(groesse, "run_ffmpeg", lambda a, **k: aufrufe.append(a))
    p = _datei(tmp_path, 1000)
    assert groesse.einhalten(p, max_bytes=2000, dauer_s=10) == 1000
    assert aufrufe == []


def _ffmpeg_schreibt(groessen: list[int], aufrufe: list[list[str]]):
    def fake(args: list[str], **_k: Any) -> None:
        aufrufe.append(args)
        Path(args[-1]).write_bytes(b"\0" * groessen[len(aufrufe) - 1])
    return fake


def test_zu_gross_wird_neu_kodiert(tmp_path: Path, monkeypatch: Any) -> None:
    aufrufe: list[list[str]] = []
    monkeypatch.setattr(groesse, "run_ffmpeg", _ffmpeg_schreibt([900], aufrufe))
    p = _datei(tmp_path, 5000)
    assert groesse.einhalten(p, max_bytes=1000, dauer_s=10) == 900
    assert p.stat().st_size == 900 and len(aufrufe) == 1
    a = aufrufe[0]
    assert a[a.index("-b:v") + 1].endswith("k") and "-maxrate" in a and "-bufsize" in a
    assert a[a.index("-c:a") + 1] == "aac" and "+faststart" in a
    assert not list(tmp_path.glob("*.klein.mp4"))       # Zwischendatei weg


def test_zweiter_versuch_mit_weniger_bitrate(tmp_path: Path, monkeypatch: Any) -> None:
    aufrufe: list[list[str]] = []
    monkeypatch.setattr(groesse, "run_ffmpeg", _ffmpeg_schreibt([1500, 800], aufrufe))
    p = _datei(tmp_path, 5000)
    assert groesse.einhalten(p, max_bytes=1000, dauer_s=100) == 800
    b1 = int(aufrufe[0][aufrufe[0].index("-b:v") + 1][:-1])
    b2 = int(aufrufe[1][aufrufe[1].index("-b:v") + 1][:-1])
    assert b2 == max(300, int(b1 * 0.8))


def test_bleibt_zu_gross_fehler_mit_grund(tmp_path: Path, monkeypatch: Any) -> None:
    aufrufe: list[list[str]] = []
    monkeypatch.setattr(groesse, "run_ffmpeg", _ffmpeg_schreibt([1500, 1400], aufrufe))
    p = _datei(tmp_path, 5000)
    with pytest.raises(groesse.ZuGross, match="1400.*1000"):
        groesse.einhalten(p, max_bytes=1000, dauer_s=100)
    assert p.stat().st_size == 5000                     # Original unangetastet
    assert not list(tmp_path.glob("*.klein.mp4"))


def test_modell_max_bytes() -> None:
    assert ReelRenderRequest().max_bytes is None
    assert ReelRenderRequest(max_bytes=15_000_000).max_bytes == 15_000_000
    with pytest.raises(ValidationError):
        ReelRenderRequest(max_bytes=1000)


# --- Render-Job --------------------------------------------------------------

def _job(tmp_path: Path, monkeypatch: Any, options: dict[str, Any]) -> tuple[Any, Any]:
    from laura.config import Settings
    from laura.db import repos
    from laura.db.database import SqliteDatabase
    from laura.jobs import JobRunner, default_registry, enqueue
    from laura.render import handlers as render_handlers

    settings = Settings(workspace_root=tmp_path / "ws", start_runner=False)
    db = SqliteDatabase(settings.db_path)
    db.migrate()
    proot = settings.workspace_root / "project"
    proot.mkdir(parents=True, exist_ok=True)
    project = repos.create_project(db, name="t", rate_num=30, rate_den=1, drop_frame=False,
                                   workspace_root=str(proot))
    a = repos.create_asset(db, project_id=project["id"], type="video", display_name="v.mp4",
                           source_path=str(tmp_path / "v.mp4"))
    tl = repos.create_timeline(db, project_id=project["id"], name="cut", kind="rough_cut")
    repos.add_timeline_clip(db, timeline_id=tl["id"], asset_id=a["id"], src_in_frame=0,
                            src_out_frame_exclusive=900, seq_in_frame=0, seq_out_frame_exclusive=900)
    monkeypatch.setattr(render_handlers, "render_clips_mp4", lambda *a, **k: None)
    monkeypatch.setattr(render_handlers, "assert_or_fix_media_sync", lambda *a, **k: None)
    monkeypatch.setattr(render_handlers.os.path, "getsize", lambda p: 20_000_000)
    exp = repos.create_export(db, project_id=project["id"], timeline_id=tl["id"], format="mp4",
                              options=options)
    registry = default_registry()
    render_handlers.register_render_handlers(registry)
    runner = JobRunner(db, registry)
    enqueue(db, queue="export", kind="export.render", payload={"export_id": exp["id"]},
            idempotency_key=f"render:{exp['id']}")
    for _ in range(20):
        if not runner.run_once():
            break
    return db, repos.get_export(db, exp["id"])


def test_job_haelt_grenze_ein(tmp_path: Path, monkeypatch: Any) -> None:
    gesehen: dict[str, Any] = {}

    def fake(datei: Path, *, max_bytes: int, dauer_s: float) -> int:
        gesehen.update(max_bytes=max_bytes, dauer_s=dauer_s)
        return 14_000_000
    monkeypatch.setattr(groesse, "einhalten", fake)
    _db, exp = _job(tmp_path, monkeypatch, {"vertical": True, "max_bytes": 15_000_000})
    assert gesehen == {"max_bytes": 15_000_000, "dauer_s": 30.0}
    assert exp["status"] == "ready" and exp["size_bytes"] == 14_000_000


def test_job_ohne_grenze_unveraendert(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(groesse, "einhalten", lambda *a, **k: pytest.fail("darf nicht laufen"))
    _db, exp = _job(tmp_path, monkeypatch, {"vertical": True})
    assert exp["status"] == "ready" and exp["size_bytes"] == 20_000_000


def test_job_zu_gross_wird_fehler_mit_grund(tmp_path: Path, monkeypatch: Any) -> None:
    def fake(*_a: Any, **_k: Any) -> int:
        raise groesse.ZuGross("Video bleibt mit 16000000 Bytes ueber der Grenze von 15000000 Bytes")
    monkeypatch.setattr(groesse, "einhalten", fake)
    _db, exp = _job(tmp_path, monkeypatch, {"vertical": True, "max_bytes": 15_000_000})
    assert exp["status"] != "ready" and "Grenze" in (exp["error"] or "")
