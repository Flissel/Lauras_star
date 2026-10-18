"""Untertitelzeilen brechen am Satzende (Video-Claw Phase F, 18.10.2026).

Probe-Reel 1 zeigte Zeilen wie "Stimme. Die Untertitel" und "Hinweis auf die Hinaus": die Gruppierung
trennte nur nach Wortzahl, Breite und Pause - ein Satzende ohne Pause lief in die naechste Zeile.
"""
from __future__ import annotations

from laura.render.captions import group_caption_lines


def _w(texte: list[str]) -> list[tuple[str, int, int]]:
    return [(t, i * 10, i * 10 + 9) for i, t in enumerate(texte)]   # luecken- und pausenfrei


def _zeilen(texte: list[str]) -> list[str]:
    return [" ".join(w[0] for w in z) for z in group_caption_lines(_w(texte))]


def test_satzende_beginnt_neue_zeile() -> None:
    assert _zeilen(["deutschen", "Stimme.", "Die", "Untertitel", "laufen", "mit."]) == [
        "deutschen Stimme.", "Die Untertitel laufen mit."]


def test_frage_und_ausruf() -> None:
    assert _zeilen(["Warum?", "Darum!", "Gut"]) == ["Warum?", "Darum!", "Gut"]


def test_ohne_satzende_wie_bisher() -> None:
    assert _zeilen(["eins", "zwei", "drei", "vier", "fuenf"]) == ["eins zwei drei vier", "fuenf"]


def test_abkuerzung_mit_punkt_zaehlt_nicht_als_satzende_wenn_klein_weiter() -> None:
    """'z.B. ein' bleibt zusammen: ein Punkt vor einem klein geschriebenen Wort ist kein Satzende."""
    assert _zeilen(["z.B.", "ein", "Test."]) == ["z.B. ein Test."]
