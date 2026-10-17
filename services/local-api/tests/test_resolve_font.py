"""resolve_font: erste vorhandene Schrift statt fest Windows-Arial (17.10.2026).

Auf der Linux-VM ist LAURA_FONT nicht gesetzt; der alte feste Windows-Pfad liess
jeden drawtext (Hook, KI-Hinweis) scheitern.
"""
from __future__ import annotations

import os
from typing import Any

from laura.render import reel


def test_env_gewinnt(monkeypatch: Any) -> None:
    monkeypatch.setenv("LAURA_FONT", "C:/Fonts/x.ttf")
    assert reel.resolve_font() == r"C\\:/Fonts/x.ttf"


def test_linux_pfad_ohne_escape(monkeypatch: Any) -> None:
    monkeypatch.setenv("LAURA_FONT", "/usr/share/fonts/a.ttf")
    assert reel.resolve_font() == "/usr/share/fonts/a.ttf"


def test_erste_vorhandene(monkeypatch: Any) -> None:
    monkeypatch.delenv("LAURA_FONT", raising=False)
    dejavu = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    monkeypatch.setattr(os.path, "isfile", lambda p: p == dejavu)
    assert reel.resolve_font() == dejavu


def test_nichts_vorhanden_alter_wert(monkeypatch: Any) -> None:
    monkeypatch.delenv("LAURA_FONT", raising=False)
    monkeypatch.setattr(os.path, "isfile", lambda p: False)
    assert reel.resolve_font() == r"C\\:/Windows/Fonts/arialbd.ttf"
