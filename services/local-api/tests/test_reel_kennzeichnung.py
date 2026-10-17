"""KI-Kennzeichnung nach Art des KI-Anteils (Video-Claw, 17.10.2026).

``kennzeichnung="auto"`` am Reel-Render: der Hinweis richtet sich danach, WAS im Video
kuenstlich ist - "Stimme: KI", "Gesicht: KI", "KI · synthetisch" - und entfaellt, wenn
nichts kuenstlich ist. Ohne das Feld (Vorgabe "fest") bleibt alles wie bisher.

Nebenbei geschlossen: eine KI-Stimme, die nur UNTER den Originalton gemischt wird
(mix_mode "mix"/"duck"), galt der alten Pruefung nicht als synthetisch. Im Auto-Modus
zaehlt jedes Tonstueck aus einem synthetischen Asset.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from laura.api.models import ReelRenderRequest
from laura.config import Settings
from laura.db import repos
from laura.db.database import SqliteDatabase
from laura.jobs import JobRunner, default_registry, enqueue
from laura.main import create_app
from laura.render import handlers as render_handlers
from laura.render.mp4 import _DEFAULT_DISCLOSURE
from laura.render.reel import kennzeichnung_text


# --- reine Abbildung ---------------------------------------------------------

@pytest.mark.parametrize("arten, text", [
    (set(), None),
    ({"stimme"}, "Stimme: KI"),
    ({"gesicht"}, "Gesicht: KI"),
    ({"stimme", "gesicht"}, "Gesicht und Stimme: KI"),
    ({"synthetisch"}, "KI · synthetisch"),
    ({"synthetisch", "stimme"}, "KI · synthetisch"),
])
def test_kennzeichnung_text(arten: set[str], text: str | None) -> None:
    assert kennzeichnung_text(arten) == text


# --- Modell ------------------------------------------------------------------

def test_modell_vorgabe_fest() -> None:
    assert ReelRenderRequest().kennzeichnung == "fest"


def test_modell_auto() -> None:
    assert ReelRenderRequest(kennzeichnung="auto").kennzeichnung == "auto"


def test_modell_unbekannt() -> None:
    with pytest.raises(ValidationError):
        ReelRenderRequest(kennzeichnung="keine")  # type: ignore[arg-type]


# --- Erkennung der Arten -----------------------------------------------------

def _db(tmp_path: Path) -> tuple[Any, Any, Any]:
    settings = Settings(workspace_root=tmp_path / "ws", start_runner=False)
    db = SqliteDatabase(settings.db_path)
    db.migrate()
    proot = settings.workspace_root / "project"
    proot.mkdir(parents=True, exist_ok=True)
    project = repos.create_project(db, name="t", rate_num=30, rate_den=1, drop_frame=False,
                                   workspace_root=str(proot))
    tl = repos.create_timeline(db, project_id=project["id"], name="cut", kind="rough_cut")
    return db, project, tl


def _video(db: Any, project: Any, tl: Any, tmp_path: Path, *, synthetic: bool = False,
           ai_effect: str | None = None, role: str = "base") -> None:
    a = repos.create_asset(db, project_id=project["id"], type="video", display_name="v.mp4",
                           source_path=str(tmp_path / "v.mp4"), synthetic=synthetic,
                           ai_effect=ai_effect)
    repos.add_timeline_clip(db, timeline_id=tl["id"], asset_id=a["id"], src_in_frame=0,
                            src_out_frame_exclusive=30, seq_in_frame=0, seq_out_frame_exclusive=30,
                            **({"role": role, "lane": 1} if role != "base" else {}))


def _ton(db: Any, project: Any, tl: Any, tmp_path: Path, *, synthetic: bool,
         ai_effect: str | None, mix_mode: str = "mix") -> None:
    a = repos.create_asset(db, project_id=project["id"], type="audio", display_name="t.wav",
                           source_path=str(tmp_path / "t.wav"), synthetic=synthetic,
                           ai_effect=ai_effect)
    repos.add_timeline_audio_clip(db, timeline_id=tl["id"], asset_id=a["id"], seq_in_frame=0,
                                  seq_out_frame_exclusive=30, mix_mode=mix_mode)


def test_arten_echtes_material(tmp_path: Path) -> None:
    db, project, tl = _db(tmp_path)
    _video(db, project, tl, tmp_path)
    assert render_handlers._synthetic_kinds(db, tl["id"]) == set()


def test_arten_ki_stimme_untergemischt(tmp_path: Path) -> None:
    db, project, tl = _db(tmp_path)
    _video(db, project, tl, tmp_path)
    _ton(db, project, tl, tmp_path, synthetic=True, ai_effect="voiceover", mix_mode="mix")
    assert render_handlers._synthetic_kinds(db, tl["id"]) == {"stimme"}


def test_arten_echte_musik_ersetzt_original(tmp_path: Path) -> None:
    """Wie die alte Pruefung: ersetzter Originalton zaehlt als synthetisch."""
    db, project, tl = _db(tmp_path)
    _video(db, project, tl, tmp_path)
    _ton(db, project, tl, tmp_path, synthetic=False, ai_effect=None, mix_mode="replace_original")
    assert render_handlers._synthetic_kinds(db, tl["id"]) == {"synthetisch"}


@pytest.mark.parametrize("effekt, art", [("reenact", "gesicht"), ("lipsync", "gesicht"),
                                         ("generate_video", "synthetisch"), (None, "synthetisch")])
def test_arten_ki_video(tmp_path: Path, effekt: str | None, art: str) -> None:
    db, project, tl = _db(tmp_path)
    _video(db, project, tl, tmp_path, synthetic=True, ai_effect=effekt)
    assert render_handlers._synthetic_kinds(db, tl["id"]) == {art}


# --- Render-Job --------------------------------------------------------------

def _render(db: Any, project: Any, tl: Any, options: dict[str, Any], monkeypatch: Any) -> dict:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(render_handlers, "render_clips_mp4", lambda *a, **k: captured.update(k))
    monkeypatch.setattr(render_handlers, "assert_or_fix_media_sync", lambda *a, **k: None)
    monkeypatch.setattr(render_handlers.os.path, "getsize", lambda p: 1)
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
    return captured


def test_job_auto_mit_ki_stimme(tmp_path: Path, monkeypatch: Any) -> None:
    db, project, tl = _db(tmp_path)
    _video(db, project, tl, tmp_path)
    _ton(db, project, tl, tmp_path, synthetic=True, ai_effect="voiceover")
    k = _render(db, project, tl, {"vertical": True, "disclosure_text": None, "kennzeichnung": "auto"},
                monkeypatch)
    assert k["disclosure_text"] == "Stimme: KI"


def test_job_auto_echtes_material_ohne_hinweis(tmp_path: Path, monkeypatch: Any) -> None:
    db, project, tl = _db(tmp_path)
    _video(db, project, tl, tmp_path)
    k = _render(db, project, tl, {"vertical": True, "disclosure_text": "KI · synthetisch",
                                  "kennzeichnung": "auto"}, monkeypatch)
    assert k["disclosure_text"] is None


def test_job_fest_bleibt_wie_bisher(tmp_path: Path, monkeypatch: Any) -> None:
    db, project, tl = _db(tmp_path)
    _video(db, project, tl, tmp_path)
    k = _render(db, project, tl, {"vertical": True, "disclosure_text": _DEFAULT_DISCLOSURE},
                monkeypatch)
    assert k["disclosure_text"] == _DEFAULT_DISCLOSURE


# --- API ---------------------------------------------------------------------

def _client(tmp_path: Path) -> tuple[TestClient, Any]:
    app = create_app(Settings(workspace_root=tmp_path, token=None, start_runner=False))
    return TestClient(app), app.state.db


def test_api_auto_speichert_kennzeichnung(tmp_path: Path) -> None:
    client, db = _client(tmp_path)
    pid = client.post("/projects", json={"name": "p", "sequence_rate_num": 30,
                                         "sequence_rate_den": 1}).json()["id"]
    tl = repos.create_timeline(db, project_id=pid, name="cut", kind="rough_cut")
    r = client.post(f"/timelines/{tl['id']}/render-reel", json={"kennzeichnung": "auto"})
    assert r.status_code == 202
    opts = repos.get_export(db, r.json()["export_id"])["options"]
    assert opts["kennzeichnung"] == "auto" and opts["disclosure_text"] is None


def test_api_ohne_feld_unveraendert(tmp_path: Path) -> None:
    client, db = _client(tmp_path)
    pid = client.post("/projects", json={"name": "p", "sequence_rate_num": 30,
                                         "sequence_rate_den": 1}).json()["id"]
    tl = repos.create_timeline(db, project_id=pid, name="cut", kind="rough_cut")
    r = client.post(f"/timelines/{tl['id']}/render-reel", json={})
    opts = repos.get_export(db, r.json()["export_id"])["options"]
    assert "kennzeichnung" not in opts and opts["disclosure_text"] == "KI · synthetisch"
