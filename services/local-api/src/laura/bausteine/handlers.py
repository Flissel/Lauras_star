"""Jobs ``baustein.standbild`` und ``baustein.titelkarte`` (Video-Claw, 17.10.2026).

Beide rendern mit ffmpeg einen 1080x1920-Clip mit stiller Stereo-Tonspur (damit er sich
mit tonfuehrenden Clips ohne Sonderfall verketten laesst) und legen ihn als normales
Video-Asset mit Probe-Daten an - narrated-reel braucht die Dauer des Assets.
Kein KI-Inhalt: die Assets sind NICHT synthetisch markiert.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from ..db import repos
from ..ingest.ffmpeg import run_ffmpeg
from ..ingest.probe import probe_asset, sha256_file
from ..jobs.runner import JobContext, JobHandler
from ..render import schriften
from ..render.reel import _drawtext_pfad, resolve_font
from ..util import new_id
from .filter import farbe_ffmpeg, standbild_graph, titelkarte_graph, umbrechen

BREITE, HOEHE = 1080, 1920
_STILLE = ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
_KODIERUNG = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
              "-movflags", "+faststart"]


def _projekt(ctx: JobContext) -> dict[str, Any]:
    project = repos.get_project(ctx.db, str(ctx.payload["project_id"]))
    if project is None:
        raise ValueError("project not found")
    return project


def _bild(ctx: JobContext, project: dict[str, Any], asset_id: str) -> dict[str, Any]:
    asset = repos.get_asset(ctx.db, asset_id)
    if asset is None or asset["project_id"] != project["id"] or asset["type"] != "image":
        raise ValueError(f"asset {asset_id} ist kein Bild dieses Projekts")
    return asset


def _frames(project: dict[str, Any], dauer_s: float) -> tuple[int, str]:
    num, den = int(project["sequence_rate_num"]), int(project["sequence_rate_den"])
    return max(1, round(float(dauer_s) * num / den)), f"{num}/{den}"


def _als_asset(ctx: JobContext, project: dict[str, Any], datei: Path, name: str) -> dict[str, Any]:
    pr = probe_asset(datei)
    sha = sha256_file(datei)
    asset = repos.create_asset(ctx.db, project_id=project["id"], type="video",
                               display_name=name, source_path=str(datei))
    repos.update_asset_probe(
        ctx.db, asset["id"], type=pr.type, duration_frames=pr.duration_frames,
        rate_num=pr.rate_num, rate_den=pr.rate_den, audio_sample_rate=pr.audio_sample_rate,
        start_timecode=pr.start_timecode, width=pr.width, height=pr.height,
        codec_video=pr.codec_video, codec_audio=pr.codec_audio, is_vfr=pr.is_vfr, sha256=sha,
    )
    repos.add_asset_file(ctx.db, asset_id=asset["id"], kind="original", path=str(datei),
                         size_bytes=os.path.getsize(datei), checksum=sha)
    return {"ok": True, "asset_id": str(asset["id"]), "duration_frames": pr.duration_frames}


def handle_standbild(ctx: JobContext) -> dict[str, Any]:
    """Payload ``{project_id, asset_id, dauer_s, bewegung}``."""
    project = _projekt(ctx)
    bild = _bild(ctx, project, str(ctx.payload["asset_id"]))
    frames, fps = _frames(project, ctx.payload["dauer_s"])
    graph = standbild_graph(str(ctx.payload["bewegung"]), frames, w=BREITE, h=HOEHE, fps=fps)
    ziel = Path(project["workspace_root"]) / "bausteine" / f"{new_id()}.mp4"
    ziel.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_ffmpeg(["-i", bild["source_path"], *_STILLE, "-filter_complex", graph,
                    "-map", "[v]", "-map", "1:a", "-frames:v", str(frames), "-r", fps,
                    *_KODIERUNG, "-shortest", str(ziel)])
        return _als_asset(ctx, project, ziel, f"Standbild: {bild['display_name']}"[:200])
    except Exception:
        ziel.unlink(missing_ok=True)
        raise


def titelkarte_schrift(p: dict[str, Any]) -> str:
    """Marke in Reels (C1): Anzeigeschrift der Marke, sonst die bisherige Schrift."""
    datei = schriften.datei(str(p.get("schrift") or ""))
    return _drawtext_pfad(str(datei).replace("\\", "/")) if datei else resolve_font()


def handle_titelkarte(ctx: JobContext) -> dict[str, Any]:
    """Payload ``{project_id, text, unterzeile?, hintergrund, textfarbe, logo_asset_id?, dauer_s}``."""
    p = ctx.payload
    project = _projekt(ctx)
    frames, fps = _frames(project, p["dauer_s"])
    dauer = frames * int(project["sequence_rate_den"]) / int(project["sequence_rate_num"])
    zeilen, groesse = umbrechen(str(p["text"]))
    logo = _bild(ctx, project, str(p["logo_asset_id"])) if p.get("logo_asset_id") else None
    kennung = new_id()
    ordner = Path(project["workspace_root"]) / "bausteine"
    arbeit = ordner / f"tmp-{kennung}"
    arbeit.mkdir(parents=True, exist_ok=True)
    ziel = ordner / f"{kennung}.mp4"
    try:
        dateien = []
        for i, zeile in enumerate(zeilen):
            (arbeit / f"z{i}.txt").write_text(zeile, encoding="utf-8")
            dateien.append(f"z{i}.txt")
        unter = None
        if str(p.get("unterzeile") or "").strip():
            (arbeit / "u.txt").write_text(str(p["unterzeile"]).strip(), encoding="utf-8")
            unter = "u.txt"
        graph = titelkarte_graph(dateien, schriftgroesse=groesse, font=titelkarte_schrift(p),
                                 textfarbe=str(p.get("textfarbe") or "#ffffff"),
                                 unterzeile_datei=unter, mit_logo=logo is not None,
                                 w=BREITE, h=HOEHE)
        eingaenge = ["-f", "lavfi", "-i",
                     f"color=c={farbe_ffmpeg(str(p['hintergrund']))}:s={BREITE}x{HOEHE}:r={fps}:d={dauer}"]
        if logo is not None:
            eingaenge += ["-i", logo["source_path"]]
        ton = "2:a" if logo is not None else "1:a"
        run_ffmpeg([*eingaenge, *_STILLE, "-filter_complex", graph, "-map", "[v]", "-map", ton,
                    "-frames:v", str(frames), "-r", fps, *_KODIERUNG, "-shortest", str(ziel)],
                   cwd=arbeit)
        return _als_asset(ctx, project, ziel, f"Titelkarte: {zeilen[0]}"[:200])
    except Exception:
        ziel.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(arbeit, ignore_errors=True)


def register_bausteine_handlers(registry: dict[str, JobHandler]) -> None:
    registry["baustein.standbild"] = handle_standbild
    registry["baustein.titelkarte"] = handle_titelkarte
