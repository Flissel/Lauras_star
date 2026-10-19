"""Markenschriften fuer Reels (Video-Claw Paket C1, 19.10.2026).

Je Marke genau eine Datei, die schwerste ihres Schnitts (fette Untertitel, Hook, Titelkarten) - als TTF aus den
woff2-Dateien des Marketing-Editors umgewandelt (SIL Open Font License, Lizenz je Schrift daneben). Die IDs sind
die des Editor-Registers (vibemind-os spaces/marketing/claw/schriften.py).

libass findet eine Schrift im fontsdir ueber ihren vollen Namen (name ID 4); die Familiennamen dieser aus
variablen Schriften geschnittenen Dateien sind uneinheitlich ("Poppins SemiBold" neben "Poppins") und taugen
dafuer nicht. drawtext bekommt die Datei direkt.
"""
from __future__ import annotations

import json
from pathlib import Path

ORDNER = Path(__file__).resolve().parent.parent / "assets" / "schriften"
REGISTER: dict[str, dict] = json.loads((ORDNER / "schriften.json").read_text(encoding="utf-8"))


def datei(sid: str) -> Path | None:
    eintrag = REGISTER.get(sid) if isinstance(sid, str) else None
    return ORDNER / eintrag["datei"] if eintrag else None


def voller_name(sid: str) -> str | None:
    eintrag = REGISTER.get(sid) if isinstance(sid, str) else None
    return eintrag["voller_name"] if eintrag else None
