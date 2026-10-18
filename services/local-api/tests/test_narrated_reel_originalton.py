"""Beats mit Originalton und gemischte Untertitel (Video-Claw Phase F, 18.10.2026).

Eine Person im Video spricht selbst: der Beat bekommt kein Voiceover, der Clip laeuft genau
src_in..src_out mit seinem eigenen Ton. Untertitel "gemischt": Voiceover-Text fuer eingesprochene
Beats, Transkript fuer die Originalton-Abschnitte.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from laura.api.models import NarratedReelBeat
from laura.db import repos
from laura.db.database import Database
from laura.render.handlers import gemischte_caption_words

from .test_narrated_reel import _beat, _patch_queued_backend, _seed_project, _seed_video_asset


# --- Modell -------------------------------------------------------------------

def test_beat_originalton_braucht_ende() -> None:
    with pytest.raises(ValidationError):
        NarratedReelBeat(asset_id="a", src_in_frame=10, originalton=True)
    with pytest.raises(ValidationError):
        NarratedReelBeat(asset_id="a", src_in_frame=10, originalton=True, src_out_frame=10)
    b = NarratedReelBeat(asset_id="a", src_in_frame=10, originalton=True, src_out_frame=100)
    assert b.text == ""


def test_beat_ohne_originalton_braucht_text_und_kein_ende() -> None:
    with pytest.raises(ValidationError):
        NarratedReelBeat(asset_id="a", src_in_frame=0)
    with pytest.raises(ValidationError):
        NarratedReelBeat(asset_id="a", src_in_frame=0, text="x", src_out_frame=50)


# --- gemischte Untertitel (rein) ------------------------------------------------

def test_gemischte_woerter() -> None:
    vo = [("Titel.", 0, 20)]
    tr = [("unter", 5, 10), ("Hallo", 40, 50), ("Welt.", 51, 60)]
    w = gemischte_caption_words(vo, tr, [(0, 30)])
    assert [x[0] for x in w] == ["Titel.", "Hallo", "Welt."]     # Transkript unter dem Voiceover faellt weg


# --- Ende-zu-Ende ueber API + Job ------------------------------------------------

def _lauf(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, beats_fn,
          frames: list[int]) -> tuple[Database, dict[str, Any], str]:
    app = cast(Any, client.app)
    db: Database = app.state.db
    project = _seed_project(db, tmp_path)
    asset = _seed_video_asset(db, project, tmp_path, name="a", duration_frames=1000)
    _patch_queued_backend(monkeypatch, frames)
    r = client.post(f"/projects/{project['id']}/narrated-reel",
                    json={"beats": beats_fn(asset), "backend": "stub", "render": False, "crossfade_frames": 0})
    assert r.status_code == 202, r.text
    assert app.state.runner.run_once() is True
    job = repos.get_job(db, r.json()["job_id"])
    assert job is not None and job["status"] == "succeeded", job
    return db, json.loads(job["result_json"]), r.json()["timeline_id"]


def test_originalton_beat_ohne_voiceover(client: TestClient, monkeypatch: pytest.MonkeyPatch,
                                         tmp_path: Path) -> None:
    db, ergebnis, tl = _lauf(client, monkeypatch, tmp_path, lambda a: [
        _beat(a, text="Einleitung."),
        {"asset_id": a["id"], "src_in_frame": 300, "src_out_frame": 450, "originalton": True},
    ], [40])
    clips = sorted(repos.list_timeline_clips(db, tl), key=lambda c: c["seq_in_frame"])
    assert len(clips) == 2
    assert (clips[1]["src_in_frame"], clips[1]["src_out_frame_exclusive"]) == (300, 450)
    ton = repos.list_timeline_audio_clips(db, tl)
    assert len(ton) == 1                                  # nur der eingesprochene Beat
    assert ton[0]["seq_out_frame_exclusive"] <= clips[1]["seq_in_frame"]
    assert ergebnis["beats"][1]["originalton"] is True and ergebnis["beats"][1]["voice_asset_id"] is None


@pytest.mark.parametrize("beat, grund", [
    ({"src_in_frame": 0, "src_out_frame": 1200, "originalton": True}, "Video-Ende"),
    ({"src_in_frame": 0, "src_out_frame": 1000, "originalton": True}, "60 s"),       # 33 s ok? 1000/30=33 s
])
def test_originalton_api_pruefungen(client: TestClient, tmp_path: Path, beat: dict, grund: str) -> None:
    app = cast(Any, client.app)
    db: Database = app.state.db
    project = _seed_project(db, tmp_path)
    asset = _seed_video_asset(db, project, tmp_path, name="a", duration_frames=1000 if grund != "60 s" else 5000)
    if grund == "60 s":
        beat = {**beat, "src_out_frame": 1900}
    r = client.post(f"/projects/{project['id']}/narrated-reel", json={"beats": [{"asset_id": asset["id"], **beat}]})
    assert r.status_code == 422 and grund in r.text


def test_alte_beats_behalten_schluessel() -> None:
    from laura.jobs.keys import idempotency_key_for
    alt = {"project_id": "p", "beats": [{"text": "a", "asset_id": "x", "src_in_frame": 0, "pad_frames": 12}]}
    neu = {"project_id": "p", "beats": [NarratedReelBeat(text="a", asset_id="x", src_in_frame=0)
                                         .model_dump(exclude={"originalton", "src_out_frame"})]}
    assert idempotency_key_for("ai.narrated_reel", alt) == idempotency_key_for("ai.narrated_reel", neu)


# --- render-reel nimmt caption_source -------------------------------------------

def test_render_reel_caption_source(client: TestClient, tmp_path: Path) -> None:
    app = cast(Any, client.app)
    db: Database = app.state.db
    project = _seed_project(db, tmp_path)
    tl = repos.create_timeline(db, project_id=project["id"], name="cut", kind="rough_cut")
    r = client.post(f"/timelines/{tl['id']}/render-reel", json={"caption_source": "gemischt"})
    assert r.status_code == 202
    assert repos.get_export(db, r.json()["export_id"])["options"]["caption_source"] == "gemischt"
    r2 = client.post(f"/timelines/{tl['id']}/render-reel", json={})
    assert "caption_source" not in repos.get_export(db, r2.json()["export_id"])["options"]
    assert client.post(f"/timelines/{tl['id']}/render-reel", json={"caption_source": "x"}).status_code == 422
