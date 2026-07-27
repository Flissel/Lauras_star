"""POST /projects/{project_id}/auto-short scouts material into a production session."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from laura.config import Settings
from laura.db import repos
from laura.db.database import SqliteDatabase
from laura.main import create_app

_TOKEN = "test-token"
_HEADERS = {"X-Laura-Token": _TOKEN}


def _app(tmp_path: Path) -> tuple[TestClient, SqliteDatabase]:
    settings = Settings(workspace_root=tmp_path / "ws", token=_TOKEN, start_runner=False)
    app = create_app(settings)
    return TestClient(app), app.state.db


def _seed_project(db: SqliteDatabase) -> str:
    project = repos.create_project(
        db, name="p", rate_num=30, rate_den=1, drop_frame=False, workspace_root="/tmp/p"
    )
    return str(project["id"])


def _seed_asset_with_material(db: SqliteDatabase, project_id: str) -> str:
    asset = repos.create_asset(
        db,
        project_id=project_id,
        type="video",
        display_name="mission.mp4",
        source_path="/tmp/mission.mp4",
    )
    asset_id = str(asset["id"])
    run = repos.create_analysis_run(db, asset_id=asset_id, pipeline_version="test", config={})
    repos.start_analysis_run(db, str(run["id"]))
    repos.insert_segment_with_words(
        db,
        asset_id=asset_id,
        run_id=str(run["id"]),
        speaker_id=None,
        segment={
            "start_sample": 16_000,
            "end_sample": 96_000,
            "start_frame": 10,
            "end_frame": 60,
            "text": "mission briefing",
            "confidence": 1.0,
        },
        words=[],
    )
    repos.finish_analysis_run(db, str(run["id"]), status="succeeded", diagnostics={})
    timeline = repos.create_timeline(
        db,
        project_id=project_id,
        name="Rough Cut",
        kind="rough_cut",
        created_from=asset_id,
    )
    repos.add_timeline_clip(
        db,
        timeline_id=str(timeline["id"]),
        asset_id=asset_id,
        src_in_frame=0,
        src_out_frame_exclusive=300,
        seq_in_frame=0,
        seq_out_frame_exclusive=300,
    )
    repos.replace_scenes(db, project_id, str(timeline["id"]), [(0, 300)])
    return asset_id


def _post(client: TestClient, project_id: str) -> Any:
    return client.post(
        f"/projects/{project_id}/auto-short",
        json={"topic": "mission", "target_seconds": 45, "format": "x", "language": "English"},
        headers=_HEADERS,
    )


def _session_count(db: SqliteDatabase) -> int:
    with db.connection() as conn:
        row = conn.execute("SELECT COUNT(*) AS count FROM production_sessions").fetchone()
    assert row is not None
    return int(row["count"])


def test_auto_short_scouts_material_and_creates_a_production_session(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr("laura.api.short_creator._autoshort_available", lambda: True)
    monkeypatch.setattr("laura.short_creator.discovery.get_index", lambda: None)
    monkeypatch.delenv("LAURA_AGENT_PROVIDER", raising=False)
    client, db = _app(tmp_path)
    project_id = _seed_project(db)
    asset_id = _seed_asset_with_material(db, project_id)

    def fixed_scout(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "asset_id": asset_id,
            "scene_numbers": [1],
            "rationale": "strongest opening",
            "fallback": False,
        }

    monkeypatch.setattr("laura.api.short_creator.run_scout", fixed_scout)

    response = _post(client, project_id)

    assert response.status_code == 202, response.text
    body = response.json()
    assert body["asset_id"] == asset_id
    assert body["scene_numbers"] == [1]
    assert body["rationale"] == "strongest opening"
    assert body["fallback"] is False
    assert body["ranking"][0]["asset_id"] == asset_id
    assert isinstance(body["warnings"], list)

    session = repos.get_production_session(db, body["session_id"])
    assert session is not None
    assert session["asset_id"] == asset_id
    job = repos.get_job(db, body["job_id"])
    assert job is not None
    payload = json.loads(job["payload_json"])
    assert payload == {
        "asset_id": asset_id,
        "session_id": body["session_id"],
        "task": (
            "mission\n\nMaterial scout: use asset 'mission.mp4'. Focus on scenes 1 — "
            "transcript hits: mission briefing. Scout rationale: strongest opening"
        ),
        "target_seconds": 45,
        "format": "x",
        "language": "English",
    }


def test_auto_short_surfaces_a_fallback_scout_decision(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr("laura.api.short_creator._autoshort_available", lambda: True)
    monkeypatch.setattr("laura.short_creator.discovery.get_index", lambda: None)
    client, db = _app(tmp_path)
    project_id = _seed_project(db)
    asset_id = _seed_asset_with_material(db, project_id)
    monkeypatch.setattr(
        "laura.api.short_creator.run_scout",
        lambda *args, **kwargs: {
            "asset_id": asset_id,
            "scene_numbers": [1],
            "rationale": "automatic fallback: top search score",
            "fallback": True,
        },
    )

    response = _post(client, project_id)

    assert response.status_code == 202, response.text
    assert response.json()["fallback"] is True


def test_auto_short_returns_422_without_material_before_creating_a_session(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr("laura.api.short_creator._autoshort_available", lambda: True)
    monkeypatch.setattr("laura.short_creator.discovery.get_index", lambda: None)
    client, db = _app(tmp_path)
    project_id = _seed_project(db)

    response = _post(client, project_id)

    assert response.status_code == 422, response.text
    assert response.json()["detail"]["reason"] == "no material found for topic"
    assert _session_count(db) == 0


def test_auto_short_returns_404_for_an_unknown_project(tmp_path: Path) -> None:
    client, _db = _app(tmp_path)

    response = _post(client, "does-not-exist")

    assert response.status_code == 404, response.text


def test_auto_short_preflight_fails_before_scout_or_session(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr("laura.api.short_creator._autoshort_available", lambda: True)
    monkeypatch.setenv("LAURA_AGENT_PROVIDER", "openai-compat")
    monkeypatch.setenv("LAURA_AGENT_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.delenv("LAURA_AGENT_API_KEY", raising=False)
    client, db = _app(tmp_path)
    project_id = _seed_project(db)
    _seed_asset_with_material(db, project_id)
    scout_called = False

    def forbidden_scout(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal scout_called
        scout_called = True
        raise AssertionError("scout must not run when provider preflight fails")

    monkeypatch.setattr("laura.api.short_creator.run_scout", forbidden_scout)

    response = _post(client, project_id)

    assert response.status_code == 503, response.text
    assert "LAURA_AGENT_API_KEY" in response.text
    assert scout_called is False
    assert _session_count(db) == 0
