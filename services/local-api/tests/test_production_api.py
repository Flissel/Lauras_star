"""POST /assets/{asset_id}/production — creates a v2 production session + enqueues production.run.
POST /production/{session_id}/message — follow-up message on an existing session's board.
GET /production/{session_id} — read-only board status + resume point.

Mirrors test_shorts_render_api.py's app-factory + token-header pattern. The job is NOT drained
here (no real LLM/orchestrator call) — we assert the session row and the queued job's kind +
payload instead. ``_autoshort_available`` is monkeypatched so tests never depend on whether the
optional 'autoshort' extra is actually installed (see test_short_creator_api.py).

The message/status tests need a REAL board on disk (``Board.create``), so unlike ``_seed_asset``
(a fake, never-written ``workspace_root="/tmp/p"`` — fine when only DB rows are asserted), the
board-backed fixture below gives the project a real ``workspace_root`` under ``tmp_path``, mirroring
``tests/test_production_orchestrator.py``'s ``_seed_scene`` convention.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from laura.config import Settings
from laura.db import repos
from laura.db.database import SqliteDatabase
from laura.main import create_app
from laura.short_creator.board import Board
from laura.short_creator.board_models import BoardMeta, ContactSheet, ContactSheetTile
from laura.short_creator.production_orchestrator import board_root_for

_TOKEN = "test-token"
_H = {"X-Laura-Token": _TOKEN}


def _app(tmp_path: Path) -> tuple[TestClient, SqliteDatabase]:
    settings = Settings(workspace_root=tmp_path / "ws", token=_TOKEN, start_runner=False)
    app = create_app(settings)
    db: SqliteDatabase = app.state.db
    return TestClient(app), db


def _seed_asset(db: SqliteDatabase) -> tuple[str, str]:
    """Project + one video asset. Returns (asset_id, project_id)."""
    project = repos.create_project(
        db, name="p", rate_num=30, rate_den=1, drop_frame=False, workspace_root="/tmp/p"
    )
    asset = repos.create_asset(
        db, project_id=project["id"], type="video",
        display_name="a", source_path="/tmp/a.mp4",
    )
    return asset["id"], project["id"]


def _seed_session_with_board(
    db: SqliteDatabase,
    tmp_path: Path,
    *,
    session_id: str = "sess_001",
    task: str = "Make a 30s recap",
    target_seconds: float = 45.0,
) -> str:
    """Project (REAL workspace_root under tmp_path) + asset + production session + a board
    already created via ``Board.create``. Returns asset_id."""
    project = repos.create_project(
        db,
        name="p",
        rate_num=30,
        rate_den=1,
        drop_frame=False,
        workspace_root=str(tmp_path / "ws" / "proj"),
    )
    asset = repos.create_asset(
        db, project_id=project["id"], type="video",
        display_name="a", source_path="/tmp/a.mp4",
    )
    repos.create_production_session(
        db, session_id=session_id, asset_id=asset["id"], created_utc="2026-07-14T10:00:00Z"
    )
    root = board_root_for(db, asset["id"], session_id)
    meta = BoardMeta(
        session_id=session_id,
        asset_id=asset["id"],
        created_utc="2026-07-14T10:00:00Z",
        task=task,
        target_seconds=target_seconds,
    )
    Board.create(root, meta)
    return str(asset["id"])


def test_create_production_enqueues_job_and_creates_session(
    tmp_path: Path, monkeypatch: Any
) -> None:
    client, db = _app(tmp_path)
    monkeypatch.setattr("laura.api.short_creator._autoshort_available", lambda: True)
    asset_id, _project_id = _seed_asset(db)

    r = client.post(
        f"/assets/{asset_id}/production",
        json={"task": "Make a 30s recap", "target_seconds": 45},
        headers=_H,
    )
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["session_id"]
    assert body["job_id"]

    # Session row exists.
    session = repos.get_production_session(db, body["session_id"])
    assert session is not None
    assert session["asset_id"] == asset_id

    # Job row exists, correct kind, payload carries session_id + the rest of the request.
    job = repos.get_job(db, body["job_id"])
    assert job is not None
    assert job["kind"] == "production.run"
    payload = json.loads(job["payload_json"])
    assert payload == {
        "asset_id": asset_id,
        "session_id": body["session_id"],
        "task": "Make a 30s recap",
        "target_seconds": 45,
    }


def test_create_production_unknown_asset_404(tmp_path: Path) -> None:
    client, _db = _app(tmp_path)
    r = client.post("/assets/does-not-exist/production", json={"task": "recap"}, headers=_H)
    assert r.status_code == 404, r.text


def test_create_production_empty_task_422(tmp_path: Path, monkeypatch: Any) -> None:
    client, db = _app(tmp_path)
    monkeypatch.setattr("laura.api.short_creator._autoshort_available", lambda: True)
    asset_id, _project_id = _seed_asset(db)

    r = client.post(f"/assets/{asset_id}/production", json={"task": ""}, headers=_H)
    assert r.status_code == 422, r.text


def test_create_production_session_row_has_correct_asset_id(
    tmp_path: Path, monkeypatch: Any
) -> None:
    client, db = _app(tmp_path)
    monkeypatch.setattr("laura.api.short_creator._autoshort_available", lambda: True)
    asset_id, _project_id = _seed_asset(db)

    r = client.post(f"/assets/{asset_id}/production", json={"task": "recap"}, headers=_H)
    assert r.status_code == 202, r.text
    session_id = r.json()["session_id"]

    session = repos.get_production_session(db, session_id)
    assert session is not None
    assert session["session_id"] == session_id
    assert session["asset_id"] == asset_id
    assert session["created_utc"]


# ---------------------------------------------------------------------------
# POST /production/{session_id}/message
# ---------------------------------------------------------------------------


def test_send_message_enqueues_job_with_board_meta_task(tmp_path: Path, monkeypatch: Any) -> None:
    client, db = _app(tmp_path)
    monkeypatch.setattr("laura.api.short_creator._autoshort_available", lambda: True)
    asset_id = _seed_session_with_board(
        db, tmp_path, session_id="sess_001", task="Make a 30s recap", target_seconds=45.0
    )

    r = client.post(
        "/production/sess_001/message",
        json={"text": "make the hook punchier"},
        headers=_H,
    )
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["session_id"] == "sess_001"
    assert body["job_id"]

    # Job payload: task/target_seconds come from the board's meta, not the request body
    # (the request only ever supplies the follow-up text).
    job = repos.get_job(db, body["job_id"])
    assert job is not None
    assert job["kind"] == "production.run"
    payload = json.loads(job["payload_json"])
    assert payload == {
        "asset_id": asset_id,
        "session_id": "sess_001",
        "task": "Make a 30s recap",
        "target_seconds": 45,
        "message": "make the hook punchier",
    }


def test_send_message_unknown_session_404(tmp_path: Path) -> None:
    client, _db = _app(tmp_path)
    r = client.post(
        "/production/does-not-exist/message", json={"text": "go back a version"}, headers=_H
    )
    assert r.status_code == 404, r.text


def test_send_message_session_without_board_404(tmp_path: Path, monkeypatch: Any) -> None:
    client, db = _app(tmp_path)
    monkeypatch.setattr("laura.api.short_creator._autoshort_available", lambda: True)
    asset_id, _project_id = _seed_asset(db)
    repos.create_production_session(
        db, session_id="sess_no_board", asset_id=asset_id, created_utc="2026-07-14T10:00:00Z"
    )

    r = client.post(
        "/production/sess_no_board/message", json={"text": "go back a version"}, headers=_H
    )
    assert r.status_code == 404, r.text


# ---------------------------------------------------------------------------
# GET /production/{session_id}
# ---------------------------------------------------------------------------


def test_get_production_status_shape(tmp_path: Path) -> None:
    client, db = _app(tmp_path)
    _seed_session_with_board(db, tmp_path, session_id="sess_002")

    r = client.get("/production/sess_002", headers=_H)
    assert r.status_code == 200, r.text
    body = r.json()

    assert set(body) >= {"meta", "scene_reviews", "artifacts", "resume_point"}
    assert body["meta"]["session_id"] == "sess_002"
    assert body["scene_reviews"] == {"count": 0, "scenes": []}
    assert set(body["artifacts"]) == {
        "storyline", "script", "voice", "cutlist", "contact_sheet", "render_report", "qa_report",
    }
    # Fresh board, no expected scenes resolvable (no rough cut) -> first chain artifact.
    assert body["resume_point"] == "storyline"


def test_get_production_status_unknown_session_404(tmp_path: Path) -> None:
    client, _db = _app(tmp_path)
    r = client.get("/production/does-not-exist", headers=_H)
    assert r.status_code == 404, r.text


def _save_contact_sheet_on_board(
    db: SqliteDatabase, asset_id: str, session_id: str, png: Path
) -> None:
    """Seed a contact_sheet artifact (pointing at *png*) straight onto the session's board."""
    board = Board.open(board_root_for(db, asset_id, session_id))
    board.save(
        "contact_sheet",
        ContactSheet(
            png_path=str(png),
            cols=1,
            rows=1,
            tiles=[ContactSheetTile(order=0, scene_number=1, frame=30, label="0 S1")],
        ),
    )


def test_get_production_status_carries_contact_sheet_details(tmp_path: Path) -> None:
    """The artifacts block lists contact_sheet like every chain artifact (version +
    archived_versions) and — once one exists — additionally its png_path + tile list, so a
    client can show the checkpoint without another roundtrip."""
    client, db = _app(tmp_path)
    asset_id = _seed_session_with_board(db, tmp_path, session_id="sess_003")
    png = tmp_path / "sheet.png"
    png.write_bytes(b"png-bytes")
    _save_contact_sheet_on_board(db, asset_id, "sess_003", png)

    r = client.get("/production/sess_003", headers=_H)
    assert r.status_code == 200, r.text
    sheet = r.json()["artifacts"]["contact_sheet"]

    assert sheet["version"] == 1
    assert sheet["png_path"] == str(png)
    assert sheet["labeled"] is True
    assert sheet["tiles"] == [{"order": 0, "scene_number": 1, "frame": 30, "label": "0 S1"}]


# ---------------------------------------------------------------------------
# GET /production/{session_id}/contact-sheet
# ---------------------------------------------------------------------------


def test_get_contact_sheet_serves_png(tmp_path: Path) -> None:
    client, db = _app(tmp_path)
    asset_id = _seed_session_with_board(db, tmp_path, session_id="sess_004")
    png = tmp_path / "sheet.png"
    png.write_bytes(b"png-bytes")
    _save_contact_sheet_on_board(db, asset_id, "sess_004", png)

    r = client.get("/production/sess_004/contact-sheet", headers=_H)

    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/png"
    assert r.content == b"png-bytes"


def test_get_contact_sheet_404s(tmp_path: Path) -> None:
    """404 on: unknown session, board without a contact_sheet artifact yet, and an artifact
    whose png has vanished from disk (the DB/board lookup comes FIRST — the filesystem is only
    touched at a path the board's own artifact recorded, never one derived from client input)."""
    client, db = _app(tmp_path)

    r = client.get("/production/does-not-exist/contact-sheet", headers=_H)
    assert r.status_code == 404, r.text

    asset_id = _seed_session_with_board(db, tmp_path, session_id="sess_005")
    r = client.get("/production/sess_005/contact-sheet", headers=_H)
    assert r.status_code == 404, r.text

    png = tmp_path / "gone.png"
    png.write_bytes(b"png-bytes")
    _save_contact_sheet_on_board(db, asset_id, "sess_005", png)
    png.unlink()
    r = client.get("/production/sess_005/contact-sheet", headers=_H)
    assert r.status_code == 404, r.text
