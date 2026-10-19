"""Marke in Reels (Paket C1, 19.10.2026): Untertitel in der Markenschrift mit dem gesprochenen Wort in der
Akzentfarbe, Hook und Titelkarte in der Anzeigeschrift. Ohne die neuen Felder bleibt alles wie bisher.

Die Schriften liegen als TTF unter laura/assets/schriften (je Marke die schwerste Datei, SIL OFL). libass findet
sie ueber den vollen Namen (name ID 4) im fontsdir - gemessen 19.10.: "Oxanium ExtraLight Bold" ->
OxaniumExtraLight-Bold; ein unbekannter Name faellt auf Arial zurueck, der Render bricht nie ab.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from laura.config import Settings
from laura.db import repos
from laura.main import create_app
from laura.render import schriften
from laura.render.captions import build_ass, farbe_ass
from laura.render.reel import reel_video_chain

from .test_reel_groesse import _job

MARKEN = {"bodoni", "cormorant", "dm-sans", "josefin", "manrope", "montserrat", "oxanium", "playfair", "poppins",
          "rajdhani", "young-serif"}


# --- Schriftregister ---------------------------------------------------------------------------

def test_register_hat_alle_markenschriften_mit_datei() -> None:
    assert set(schriften.REGISTER) == MARKEN
    for sid in MARKEN:
        assert schriften.datei(sid).is_file(), sid
        assert schriften.voller_name(sid)
        assert (schriften.ORDNER / f"OFL-{sid}.txt").is_file(), sid


def test_unbekannte_schrift() -> None:
    assert schriften.datei("comic") is None and schriften.voller_name("comic") is None
    assert schriften.datei("../etc/passwd") is None


# --- Untertitel ----------------------------------------------------------------------------------

LINIE = [[("So", 0, 10), ("sieht", 10, 25), ("es", 25, 30)]]


def _dialoge(ass: str) -> list[str]:
    return [z for z in ass.splitlines() if z.startswith("Dialogue:")]


def test_farbe_ass() -> None:
    assert farbe_ass("#5b3fd6") == "&H00D63F5B"
    assert farbe_ass("#FFFFFF") == "&H00FFFFFF"


def test_vorgabe_unveraendert_arial_und_kf() -> None:
    ass = build_ass(LINIE, rate_num=30, rate_den=1)
    assert "Style: Reel,Arial," in ass
    assert len(_dialoge(ass)) == 1 and "\\kf" in _dialoge(ass)[0]


def test_markenschrift_im_stil() -> None:
    ass = build_ass(LINIE, rate_num=30, rate_den=1, font="Oxanium ExtraLight Bold")
    assert "Style: Reel,Oxanium ExtraLight Bold,72," in ass
    stil = [z for z in ass.splitlines() if z.startswith("Style:")][0].split(",")
    assert stil[7] == "0"     # Bold aus: die Datei ist schon die fette Fassung, kein kuenstliches Fett


def test_akzent_faerbt_genau_das_gesprochene_wort() -> None:
    ass = build_ass(LINIE, rate_num=30, rate_den=1, akzent="#5b3fd6")
    d = _dialoge(ass)
    assert len(d) == 3
    # Marker (User-Entscheid 19.10. nach dem echten Reel): Wort bleibt weiss, breite Kontur in der Akzentfarbe -
    # in reiner Akzentfarbe verschwand es auf der Akzent-Karte und auf dunklen Videos
    an = "{\\3c&HD63F5B&\\bord9}"
    aus = "{\\3c&H000000&\\bord3}"
    assert d[0].endswith(f",{an}So{aus} sieht es")
    assert d[1].endswith(f",So {an}sieht{aus} es")
    assert d[2].endswith(f",So sieht {an}es{aus}")
    # lueckenlos: jedes Wort von seinem Start bis zum Start des naechsten, das letzte bis Zeilenende
    assert [z.split(",")[1:3] for z in d] == [["0:00:00.00", "0:00:00.33"], ["0:00:00.33", "0:00:00.83"],
                                               ["0:00:00.83", "0:00:01.00"]]
    assert "\\kf" not in ass


def test_akzent_im_normalmodus_ohne_hervorhebung() -> None:
    ass = build_ass(LINIE, rate_num=30, rate_den=1, akzent="#5b3fd6", mode="normal")
    assert len(_dialoge(ass)) == 1 and "\\3c" not in ass


def test_akzent_escaped_woerter() -> None:
    ass = build_ass([[("{boese}", 0, 10), ("x", 10, 20)]], rate_num=30, rate_den=1, akzent="#000000")
    assert "{boese}" not in ass


# --- Hook und Filter -----------------------------------------------------------------------------

def test_hook_in_eigener_schrift_hinweis_bleibt() -> None:
    kette = reel_video_chain(vertical=True, hook_textfile="h.txt", disclosure_textfile="d.txt",
                             font="arial.ttf", hook_font="oxanium.ttf")
    teile = [t for t in kette.split(",") if t.startswith("drawtext")]
    assert "fontfile=oxanium.ttf" in teile[0] and "textfile=h.txt" in teile[0]
    assert "fontfile=arial.ttf" in teile[1] and "textfile=d.txt" in teile[1]


def test_hook_ohne_eigene_schrift_wie_bisher() -> None:
    assert reel_video_chain(vertical=True, hook_textfile="h.txt", font="arial.ttf") == \
        reel_video_chain(vertical=True, hook_textfile="h.txt", font="arial.ttf", hook_font=None)


def test_untertitel_filter_mit_schriftordner() -> None:
    from laura.render.mp4 import untertitel_filter
    assert untertitel_filter("c.ass", None, None) == "ass=c.ass"
    assert untertitel_filter(None, "c.srt", None) == "subtitles=c.srt"
    assert untertitel_filter(None, None, "/x") == ""
    f = untertitel_filter("c.ass", None, "/app/src/laura/assets/schriften")
    assert f == "ass=c.ass:fontsdir=/app/src/laura/assets/schriften"
    assert untertitel_filter("c.ass", None, r"C:\x\y") == r"ass=c.ass:fontsdir=C\\:/x/y"   # wie drawtext


# --- Optionen im Render-Job ----------------------------------------------------------------------

def test_untertitel_stil_aus_optionen() -> None:
    from laura.render.handlers import untertitel_stil
    assert untertitel_stil({}) == ("Arial", None, None)
    name, akzent, ordner = untertitel_stil({"caption_schrift": "oxanium", "caption_akzent": "#5b3fd6"})
    assert name == schriften.voller_name("oxanium") and akzent == "#5b3fd6" and ordner == str(schriften.ORDNER)
    assert untertitel_stil({"caption_schrift": "comic", "caption_akzent": "rot"}) == ("Arial", None, None)


def _abfangen(monkeypatch: Any, gesehen: dict) -> None:
    from laura.render import handlers as rh
    real_setattr = monkeypatch.setattr

    def setattr_abfangen(ziel: Any, name: str, wert: Any, *a: Any, **k: Any) -> None:
        if ziel is rh and name == "render_clips_mp4":
            wert = lambda *aa, **kk: gesehen.update(kk)  # noqa: E731
        real_setattr(ziel, name, wert, *a, **k)
    monkeypatch.setattr = setattr_abfangen  # type: ignore[method-assign]


def test_job_reicht_hookschrift_durch(tmp_path: Path, monkeypatch: Any) -> None:
    gesehen: dict = {}
    _abfangen(monkeypatch, gesehen)
    _job(tmp_path, monkeypatch, {"vertical": True, "hook_text": "Hallo", "hook_schrift": "oxanium"})
    assert gesehen.get("hook_font") == str(schriften.datei("oxanium"))


def test_job_ohne_markenfelder_wie_bisher(tmp_path: Path, monkeypatch: Any) -> None:
    gesehen: dict = {}
    _abfangen(monkeypatch, gesehen)
    _job(tmp_path, monkeypatch, {"vertical": True})
    assert "hook_font" not in gesehen and "caption_fontsdir" not in gesehen


# --- API ------------------------------------------------------------------------------------------

def _client(tmp_path: Path) -> tuple[TestClient, Any]:
    app = create_app(Settings(workspace_root=tmp_path, token=None, start_runner=False))
    return TestClient(app), app.state.db


def _tl(client: TestClient, db: Any) -> str:
    pid = client.post("/projects", json={"name": "p", "sequence_rate_num": 30,
                                         "sequence_rate_den": 1}).json()["id"]
    return repos.create_timeline(db, project_id=pid, name="cut", kind="rough_cut")["id"]


def test_api_speichert_markenfelder(tmp_path: Path) -> None:
    client, db = _client(tmp_path)
    tl = _tl(client, db)
    r = client.post(f"/timelines/{tl}/render-reel", json={
        "caption_schrift": "dm-sans", "caption_akzent": "#5b3fd6", "hook_schrift": "oxanium"})
    assert r.status_code == 202, r.text
    o = repos.get_export(db, r.json()["export_id"])["options"]
    assert (o["caption_schrift"], o["caption_akzent"], o["hook_schrift"]) == ("dm-sans", "#5b3fd6", "oxanium")


def test_api_ohne_markenfelder_unveraendert(tmp_path: Path) -> None:
    client, db = _client(tmp_path)
    tl = _tl(client, db)
    o = repos.get_export(db, client.post(f"/timelines/{tl}/render-reel", json={}).json()["export_id"])["options"]
    assert not {"caption_schrift", "caption_akzent", "hook_schrift"} & set(o)


@pytest.mark.parametrize("koerper", [
    {"caption_schrift": "comic"}, {"caption_schrift": "../x"}, {"hook_schrift": "arial"},
    {"caption_akzent": "rot"}, {"caption_akzent": "#12345"},
])
def test_api_formen(tmp_path: Path, koerper: dict) -> None:
    client, db = _client(tmp_path)
    tl = _tl(client, db)
    assert client.post(f"/timelines/{tl}/render-reel", json=koerper).status_code == 422


def test_titelkarte_nimmt_schrift_an_und_prueft_sie(tmp_path: Path) -> None:
    from laura.api.bausteine import TitelkarteRequest
    assert TitelkarteRequest(text="Hallo", hintergrund="#5b3fd6", schrift="oxanium").schrift == "oxanium"
    assert TitelkarteRequest(text="Hallo", hintergrund="#5b3fd6").schrift is None
    with pytest.raises(ValueError):
        TitelkarteRequest(text="Hallo", hintergrund="#5b3fd6", schrift="comic")


def test_titelkarte_rendert_mit_markenschrift(monkeypatch: Any) -> None:
    from laura.bausteine import handlers as bh
    pfad = bh.titelkarte_schrift({"schrift": "oxanium"})
    assert pfad.endswith("/assets/schriften/oxanium-700-normal.ttf") and "\\" not in pfad.replace("\\\\:", "")
    assert bh.titelkarte_schrift({}) == bh.resolve_font()
    assert bh.titelkarte_schrift({"schrift": "comic"}) == bh.resolve_font()


# --- Echter Lauf: libass waehlt die Markenschrift -------------------------------------------------

@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg fehlt")
def test_libass_waehlt_die_markenschrift(tmp_path: Path) -> None:
    from laura.render.handlers import untertitel_stil
    from laura.render.mp4 import untertitel_filter
    name, akzent, ordner = untertitel_stil({"caption_schrift": "oxanium", "caption_akzent": "#5b3fd6"})
    (tmp_path / "c.ass").write_text(build_ass(LINIE, rate_num=30, rate_den=1, font=name, akzent=akzent),
                                    encoding="utf-8")
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "debug", "-f", "lavfi", "-i",
                        "color=c=black:s=1080x1920:d=1", "-vf", untertitel_filter("c.ass", None, ordner),
                        "-frames:v", "1", "-y", "o.png"], cwd=tmp_path, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    auswahl = [z for z in r.stderr.splitlines() if "fontselect" in z]
    assert auswahl and "OxaniumExtraLight-Bold" in auswahl[0], r.stderr[-800:]
