"""Groessengrenze fuer fertige Exporte (Video-Claw, 17.10.2026).

sales-claw nimmt hoechstens 15 MB je Video. Statt in jede Encoder-Zeile einzugreifen,
kodiert ``einhalten`` eine zu grosse fertige Datei mit einer aus der Dauer berechneten
Bitrate neu (H.264 + AAC, faststart). Zweiter Versuch mit 80 %, danach ``ZuGross``.
Das Original bleibt unangetastet, bis eine passende Fassung vorliegt.
"""
from __future__ import annotations

from pathlib import Path

from ..ingest.ffmpeg import run_ffmpeg

TON_KBPS = 128
MIN_KBPS = 300
_RESERVE = 0.92          # Container-Overhead und Bitraten-Schwankung


class ZuGross(RuntimeError):
    """Auch nach dem Neukodieren groesser als erlaubt."""


def ziel_bitrate_kbps(max_bytes: int, dauer_s: float) -> int:
    if dauer_s <= 0 or max_bytes <= 0:
        raise ValueError("max_bytes und dauer_s muessen positiv sein")
    gesamt = max_bytes * 8 * _RESERVE / dauer_s / 1000
    return max(MIN_KBPS, int(gesamt - TON_KBPS))


def _kodieren(quelle: Path, ziel: Path, kbps: int) -> None:
    run_ffmpeg([
        "-i", str(quelle),
        "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-b:v", f"{kbps}k", "-maxrate", f"{kbps}k", "-bufsize", f"{kbps * 2}k",
        "-c:a", "aac", "-b:a", f"{TON_KBPS}k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart",
        str(ziel),
    ])


def einhalten(datei: Path, *, max_bytes: int, dauer_s: float) -> int:
    """Groesse von ``datei`` nach dem Einhalten der Grenze; wirft ``ZuGross``."""
    groesse = datei.stat().st_size
    if groesse <= max_bytes:
        return groesse
    kbps = ziel_bitrate_kbps(max_bytes, dauer_s)
    zwischen = datei.with_name(f"{datei.stem}.klein.mp4")
    try:
        for _versuch in range(2):
            _kodieren(datei, zwischen, kbps)
            neu = zwischen.stat().st_size
            if neu <= max_bytes:
                zwischen.replace(datei)
                return neu
            kbps = max(MIN_KBPS, int(kbps * 0.8))
        raise ZuGross(f"Video bleibt mit {neu} Bytes ueber der Grenze von {max_bytes} Bytes")
    finally:
        zwischen.unlink(missing_ok=True)
