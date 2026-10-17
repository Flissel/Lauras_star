"""Logo-Einblendung fuer fertige Reels (Video-Claw, 17.10.2026).

Ein eigener Durchlauf hinter dem Render (wie die Groessengrenze), damit der Encoder-
Pfad unberuehrt bleibt. Oben, weil unten Untertitel und KI-Hinweis sitzen; Abstand
oben so, dass die Bedienleiste der Plattformen das Logo nicht verdeckt.

Die Logogroesse wird aus den GEMESSENEN Abmessungen von Video und Logo gerechnet und
fest skaliert (scale2ref lieferte in der ffmpeg-Version am PC verzerrte Groessen).
"""
from __future__ import annotations

from pathlib import Path

from ..ingest.ffmpeg import probe, run_ffmpeg

POSITIONEN: tuple[str, ...] = ("oben_rechts", "oben_links", "oben_mitte")
_RAND_X, _RAND_Y = 48, 140


def logo_masse(video_breite: int, logo_breite: int, logo_hoehe: int,
               breite_prozent: int) -> tuple[int, int]:
    """(Breite, Hoehe) des Logos in Pixeln, gerade Zahlen, Seitenverhaeltnis erhalten."""
    if not 8 <= breite_prozent <= 25:
        raise ValueError("logo_breite_prozent muss zwischen 8 und 25 liegen")
    if min(video_breite, logo_breite, logo_hoehe) <= 0:
        raise ValueError("Abmessungen muessen positiv sein")
    lw = max(2, round(video_breite * breite_prozent / 100 / 2) * 2)
    lh = max(2, round(lw * logo_hoehe / logo_breite / 2) * 2)
    return lw, lh


def logo_graph(position: str, lw: int, lh: int) -> str:
    if position not in POSITIONEN:
        raise ValueError(f"logo_position muss eine von {', '.join(POSITIONEN)} sein")
    x = {"oben_rechts": f"W-w-{_RAND_X}", "oben_links": str(_RAND_X),
         "oben_mitte": "(W-w)/2"}[position]
    return f"[1:v]scale={lw}:{lh}[l];[0:v][l]overlay={x}:{_RAND_Y},format=yuv420p[v]"


def _breite_hoehe(pfad: Path) -> tuple[int, int]:
    stroeme = probe(pfad).get("streams", [])
    v = next(s for s in stroeme if s.get("codec_type") == "video")
    return int(v["width"]), int(v["height"])


def einblenden(datei: Path, logo: Path, *, position: str = "oben_rechts",
               breite_prozent: int = 14) -> None:
    """Logo in ``datei`` einbrennen (ersetzt die Datei erst, wenn das Ergebnis da ist)."""
    vb, _vh = _breite_hoehe(datei)
    lb, lh_quelle = _breite_hoehe(logo)
    graph = logo_graph(position, *logo_masse(vb, lb, lh_quelle, breite_prozent))
    zwischen = datei.with_name(f"{datei.stem}.logo.mp4")
    try:
        run_ffmpeg(["-i", str(datei), "-i", str(logo), "-filter_complex", graph,
                    "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "copy", "-movflags", "+faststart", str(zwischen)])
        zwischen.replace(datei)
    finally:
        zwischen.unlink(missing_ok=True)
