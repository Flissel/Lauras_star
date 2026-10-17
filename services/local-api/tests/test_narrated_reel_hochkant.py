"""narrated-reel direkt hochkant (Video-Claw, 17.10.2026).

Bisher rendert ``render=true`` immer quer (``/render`` ohne vertical) - wer ein Reel
wollte, musste render=false setzen und render-reel selbst aufrufen. ``ausgabe="reel"``
haengt den Render-Auftrag gleich mit den Reel-Optionen an: 9:16, Untertitel aus der
Stimme, Reel-Untertitelformat, Hook, KI-Hinweis nach Inhalt, optionale Groessengrenze.
Ohne das Feld bleibt alles wie bisher (auch der Idempotenz-Schluessel).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from laura.db import repos
from laura.db.database import Database

from .test_narrated_reel import _beat, _patch_queued_backend, _seed_project, _seed_video_asset


def _lauf(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
          extra: dict[str, Any]) -> dict[str, Any]:
    app = cast(Any, client.app)
    db: Database = app.state.db
    project = _seed_project(db, tmp_path)
    asset = _seed_video_asset(db, project, tmp_path, name="a", duration_frames=1000)
    _patch_queued_backend(monkeypatch, [40])
    body = {"beats": [_beat(asset, text="eins")], "backend": "stub", "render": True, **extra}
    r = client.post(f"/projects/{project['id']}/narrated-reel", json=body)
    assert r.status_code == 202, r.text
    assert app.state.runner.run_once() is True
    job = repos.get_job(db, r.json()["job_id"])
    assert job is not None and job["status"] == "succeeded", job
    export = repos.get_export(db, json.loads(job["result_json"])["export_id"])
    assert export is not None
    return dict(export["options"])


def test_ausgabe_reel_haengt_reel_render_an(client: TestClient, monkeypatch: pytest.MonkeyPatch,
                                            tmp_path: Path) -> None:
    o = _lauf(client, monkeypatch, tmp_path, {"ausgabe": "reel", "hook_text": "Drei Fehler",
                                              "max_bytes": 15_000_000})
    assert o["vertical"] is True and o["captions"] is True
    assert o["caption_source"] == "voiceover" and o["caption_preset"] == "reels"
    assert o["hook_text"] == "Drei Fehler" and o["max_bytes"] == 15_000_000
    assert o["kennzeichnung"] == "auto" and o["disclosure_text"] is None


def test_ohne_ausgabe_wie_bisher(client: TestClient, monkeypatch: pytest.MonkeyPatch,
                                 tmp_path: Path) -> None:
    o = _lauf(client, monkeypatch, tmp_path, {"caption_preset": "wide"})
    assert o == {"captions": True, "caption_source": "voiceover", "caption_preset": "wide"}


def test_reel_felder_ohne_reel_abgelehnt(client: TestClient, tmp_path: Path) -> None:
    app = cast(Any, client.app)
    db: Database = app.state.db
    project = _seed_project(db, tmp_path)
    asset = _seed_video_asset(db, project, tmp_path, name="a", duration_frames=1000)
    r = client.post(f"/projects/{project['id']}/narrated-reel",
                    json={"beats": [_beat(asset, text="eins")], "hook_text": "x"})
    assert r.status_code == 422


def test_schluessel_reel_unterscheidet_sich_von_quer() -> None:
    from laura.jobs.keys import idempotency_key_for
    basis = {"project_id": "p", "beats": [{"text": "eins"}], "render": True,
             "caption_preset": "wide"}
    quer = idempotency_key_for("ai.narrated_reel", basis)
    assert idempotency_key_for("ai.narrated_reel", {**basis, "ausgabe": "quer"}) == quer
    assert idempotency_key_for("ai.narrated_reel", {**basis, "ausgabe": "reel"}) != quer
