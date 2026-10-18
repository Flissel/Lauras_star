"""Reel-Feinschliff (Video-Claw, 18.10.2026): Rahmung, Lautheit, Untertitel-Korrektur.

Probe-Reels mit Originalton zeigten: eine breite Bildschirmaufnahme wurde durch den Mittel-Zuschnitt
unlesbar, der Ton war leise (-29 dB), und Erkennungsfehler ("Kastermann") standen in den Untertiteln.
Der Renderer konnte Einpassen (Blur-Fill) und loudnorm laengst - render-reel reichte beides nicht durch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from laura.config import Settings
from laura.db import repos
from laura.main import create_app
from laura.render.handlers import untertitel_ersetzen

from .test_reel_groesse import _job


# --- Wortersetzung (rein) ---------------------------------------------------------

def test_ersetzen_behaelt_satzzeichen() -> None:
    w = [("jeden", 0, 5), ("Kastermann,", 6, 10), ("identiert.", 11, 15), ("gut", 16, 20)]
    aus = untertitel_ersetzen(w, {"Kastermann": "Kunden", "identiert": "editiert"})
    assert [x[0] for x in aus] == ["jeden", "Kunden,", "editiert.", "gut"]
    assert [x[1:] for x in aus] == [x[1:] for x in w]          # Zeiten bleiben


def test_ersetzen_nur_ganze_woerter() -> None:
    aus = untertitel_ersetzen([("Kastermanns", 0, 5)], {"Kastermann": "Kunde"})
    assert aus[0][0] == "Kastermanns"


def test_ersetzen_leer() -> None:
    w = [("a", 0, 1)]
    assert untertitel_ersetzen(w, {}) == w and untertitel_ersetzen(w, None) == w


# --- API -----------------------------------------------------------------------------

def _client(tmp_path: Path) -> tuple[TestClient, Any]:
    app = create_app(Settings(workspace_root=tmp_path, token=None, start_runner=False))
    return TestClient(app), app.state.db


def _tl(client: TestClient, db: Any) -> str:
    pid = client.post("/projects", json={"name": "p", "sequence_rate_num": 30,
                                         "sequence_rate_den": 1}).json()["id"]
    return repos.create_timeline(db, project_id=pid, name="cut", kind="rough_cut")["id"]


def test_api_speichert_neue_felder(tmp_path: Path) -> None:
    client, db = _client(tmp_path)
    tl = _tl(client, db)
    r = client.post(f"/timelines/{tl}/render-reel", json={
        "rahmung": "einpassen", "loudnorm": True, "untertitel_ersetzen": {"Kastermann": "Kunde"}})
    assert r.status_code == 202, r.text
    o = repos.get_export(db, r.json()["export_id"])["options"]
    assert o["rahmung"] == "einpassen" and o["loudnorm"] is True
    assert o["untertitel_ersetzen"] == {"Kastermann": "Kunde"}


def test_api_ohne_felder_unveraendert(tmp_path: Path) -> None:
    client, db = _client(tmp_path)
    tl = _tl(client, db)
    o = repos.get_export(db, client.post(f"/timelines/{tl}/render-reel", json={}).json()["export_id"])["options"]
    assert not {"rahmung", "loudnorm", "untertitel_ersetzen"} & set(o)


@pytest.mark.parametrize("koerper", [
    {"rahmung": "quer"},
    {"untertitel_ersetzen": {f"w{i}": "x" for i in range(31)}},
    {"untertitel_ersetzen": {"zwei woerter": "x"}},
    {"untertitel_ersetzen": {"a": "x" * 41}},
    {"untertitel_ersetzen": {"": "x"}},
])
def test_api_formen(tmp_path: Path, koerper: dict) -> None:
    client, db = _client(tmp_path)
    tl = _tl(client, db)
    assert client.post(f"/timelines/{tl}/render-reel", json=koerper).status_code == 422


# --- Render-Job reicht durch ----------------------------------------------------------

def test_job_reicht_rahmung_und_lautheit_durch(tmp_path: Path, monkeypatch: Any) -> None:
    from laura.render import handlers as rh
    gesehen: dict = {}
    real_setattr = monkeypatch.setattr

    def setattr_abfangen(ziel: Any, name: str, wert: Any, *a: Any, **k: Any) -> None:
        if ziel is rh and name == "render_clips_mp4":
            wert = lambda *aa, **kk: gesehen.update(kk)  # noqa: E731
        real_setattr(ziel, name, wert, *a, **k)
    monkeypatch.setattr = setattr_abfangen  # type: ignore[method-assign]
    _job(tmp_path, monkeypatch, {"vertical": True, "rahmung": "einpassen", "loudnorm": True})
    assert gesehen.get("reel_blur_fill") is True and gesehen.get("loudnorm") is True


def test_job_vorgabe_ohne_einpassen_und_lautheit(tmp_path: Path, monkeypatch: Any) -> None:
    from laura.render import handlers as rh
    gesehen: dict = {}
    real_setattr = monkeypatch.setattr

    def setattr_abfangen(ziel: Any, name: str, wert: Any, *a: Any, **k: Any) -> None:
        if ziel is rh and name == "render_clips_mp4":
            wert = lambda *aa, **kk: gesehen.update(kk)  # noqa: E731
        real_setattr(ziel, name, wert, *a, **k)
    monkeypatch.setattr = setattr_abfangen  # type: ignore[method-assign]
    _job(tmp_path, monkeypatch, {"vertical": True})
    assert not gesehen.get("reel_blur_fill") and not gesehen.get("loudnorm")
