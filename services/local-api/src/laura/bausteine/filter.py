"""Reine ffmpeg-Filtergraphen fuer Standbilder und Titelkarten - kein I/O.

Text kommt wie im Reel-Renderer ueber drawtext ``textfile=`` (Basename im
Arbeitsverzeichnis von ffmpeg), nie inline: beliebiger Text laesst sich inline nicht
sicher escapen und waere ein Filtergraph-Einfallstor.
"""
from __future__ import annotations

import re

BEWEGUNGEN: tuple[str, ...] = ("ruhig", "zoom_rein", "zoom_raus", "schwenk")
_ZOOM = 0.12
_FARBE = re.compile(r"^#[0-9a-fA-F]{6}$")


def farbe_ffmpeg(hex_farbe: str) -> str:
    """'#1a2b3c' -> '0x1a2b3c'; alles andere ist ein Fehler (kein Filter-Einfallstor)."""
    if not isinstance(hex_farbe, str) or not _FARBE.fullmatch(hex_farbe):
        raise ValueError(f"Farbe muss #rrggbb sein: {hex_farbe!r}")
    return "0x" + hex_farbe[1:].lower()


def standbild_graph(bewegung: str, frames: int, *, w: int = 1080, h: int = 1920,
                    fps: str = "30/1") -> str:
    """Ein Bild (Eingang 0, EIN Frame) -> ``frames`` Frames ``w``x``h`` mit Bewegung.

    Das Bild wird NICHT beschnitten: es liegt eingepasst auf einer weichgezeichneten,
    flaechenfuellenden Kopie seiner selbst (Blur-Fill wie bei Reels ueblich). Gezoomt wird
    auf einer um 25 % groesseren Arbeitsflaeche, damit der Zoom nicht unscharf wird.
    """
    if bewegung not in BEWEGUNGEN:
        raise ValueError(f"bewegung muss eine von {', '.join(BEWEGUNGEN)} sein")
    if frames < 1:
        raise ValueError("frames muss positiv sein")
    aw, ah = (int(w * 1.25) // 2) * 2, (int(h * 1.25) // 2) * 2
    f = max(frames - 1, 1)
    mitte_x, mitte_y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    z, x, y = {
        "ruhig": ("1", mitte_x, mitte_y),
        "zoom_rein": (f"1+{_ZOOM}*on/{f}", mitte_x, mitte_y),
        "zoom_raus": (f"{1 + _ZOOM}-{_ZOOM}*on/{f}", mitte_x, mitte_y),
        "schwenk": (f"{1 + _ZOOM}", f"(iw-iw/zoom)*on/{f}", mitte_y),
    }[bewegung]
    return (
        f"[0:v]split=2[bg][fg];"
        f"[bg]scale={aw}:{ah}:force_original_aspect_ratio=increase,crop={aw}:{ah},boxblur=20:2[bl];"
        f"[fg]scale={aw}:{ah}:force_original_aspect_ratio=decrease[fl];"
        f"[bl][fl]overlay=(W-w)/2:(H-h)/2,setsar=1,"
        f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={w}x{h}:fps={fps},format=yuv420p[v]"
    )


def umbrechen(text: str) -> tuple[list[str], int]:
    """Text in hoechstens 4 Zeilen und die passende Schriftgroesse.

    Groesste Schrift zuerst; passt der Text nicht in deren Zeilenzahl, die naechst
    kleinere. Ein einzelnes zu langes Wort wird hart getrennt.
    """
    woerter = text.split()
    if not woerter:
        raise ValueError("Text ist leer")
    for groesse, breite, max_zeilen in ((96, 16, 2), (80, 19, 3), (68, 23, 4)):
        zeilen = _zeilen(woerter, breite)
        if len(zeilen) <= max_zeilen:
            return zeilen, groesse
    raise ValueError("Text passt nicht auf die Karte (hoechstens 4 Zeilen)")


def _zeilen(woerter: list[str], breite: int) -> list[str]:
    zeilen: list[str] = []
    akt = ""
    for wort in woerter:
        while len(wort) > breite:
            if akt:
                zeilen.append(akt)
                akt = ""
            zeilen.append(wort[:breite])
            wort = wort[breite:]
        if not akt:
            akt = wort
        elif len(akt) + 1 + len(wort) <= breite:
            akt = f"{akt} {wort}"
        else:
            zeilen.append(akt)
            akt = wort
    if akt:
        zeilen.append(akt)
    return zeilen


def titelkarte_graph(zeilen_dateien: list[str], *, schriftgroesse: int, font: str,
                     textfarbe: str, unterzeile_datei: str | None = None, mit_logo: bool = False,
                     w: int = 1080, h: int = 1920, einblenden_s: float = 0.4) -> str:
    """Farbflaeche (Eingang 0) + Textzeilen mittig + optional Unterzeile und Logo (Eingang 1).

    Jede Zeile ist ein eigener drawtext, damit jede fuer sich zentriert ist.
    """
    if not zeilen_dateien:
        raise ValueError("mindestens eine Zeile")
    farbe = farbe_ffmpeg(textfarbe)
    zeilenhoehe = int(schriftgroesse * 1.25)
    unter_gr = max(36, schriftgroesse // 2)
    block = zeilenhoehe * len(zeilen_dateien) + (int(unter_gr * 1.8) if unterzeile_datei else 0)
    oben = (h - block) // 2
    teile = []
    for i, datei in enumerate(zeilen_dateien):
        teile.append(f"drawtext=fontfile={font}:textfile={datei}:fontsize={schriftgroesse}"
                     f":fontcolor={farbe}:x=(w-text_w)/2:y={oben + i * zeilenhoehe}")
    if unterzeile_datei:
        y = oben + zeilenhoehe * len(zeilen_dateien) + int(unter_gr * 0.8)
        teile.append(f"drawtext=fontfile={font}:textfile={unterzeile_datei}:fontsize={unter_gr}"
                     f":fontcolor={farbe}@0.85:x=(w-text_w)/2:y={y}")
    kette = ",".join(teile)
    if mit_logo:
        logo_breite = (int(w * 0.3) // 2) * 2
        return (f"[1:v]scale={logo_breite}:-2[logo];[0:v]{kette}[t];"
                f"[t][logo]overlay=(W-w)/2:{int(h * 0.1)},fade=t=in:st=0:d={einblenden_s},"
                f"format=yuv420p[v]")
    return f"[0:v]{kette},fade=t=in:st=0:d={einblenden_s},format=yuv420p[v]"
