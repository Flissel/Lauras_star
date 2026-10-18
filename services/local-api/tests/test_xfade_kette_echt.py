"""Echte ffmpeg-Probe: drei und mehr Clips mit Ueberblendung (17.10.2026).

Gemessen auf der VM (ffmpeg 7.1): ab der zweiten Ueberblendung scheiterte jeder Render mit
"xfade: The inputs needs to be a constant frame rate; current rate of 1/0 is invalid" - die
Bildrate ging am Ausgang des vorigen xfade/settb verloren. Die Tests der Kette ersetzten
ffmpeg bisher immer; diese Probe faehrt es echt.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from laura.ingest.ffmpeg import probe, run_ffmpeg
from laura.render.mp4 import VideoTransition, render_clips_mp4


def _clip(pfad: Path, farbe: str, sekunden: int = 2) -> Path:
    run_ffmpeg(["-f", "lavfi", "-i", f"color=c={farbe}:s=320x568:r=30:d={sekunden}",
                "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-shortest",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(pfad)])
    return pfad


@pytest.mark.parametrize("anzahl", [3, 5])
def test_mehrere_ueberblendungen_rendern(tmp_path: Path, anzahl: int) -> None:
    farben = ["red", "green", "blue", "yellow", "white"]
    clips = [(_clip(tmp_path / f"c{i}.mp4", farben[i]), 0, 50) for i in range(anzahl)]
    uebergaenge = [VideoTransition(kind="crossfade", boundary_frame=50 * (i + 1), duration_frames=6)
                   for i in range(anzahl - 1)]
    ziel = tmp_path / "aus.mp4"
    render_clips_mp4(clips, ziel, rate_num=30, rate_den=1, video_transitions=uebergaenge)
    dauer = float(probe(ziel)["format"]["duration"])
    # Laura verlaengert jeden ueberblendeten Clip um die Ueberblendung (Reserve) - die Laenge
    # bleibt die Summe der Clips.
    assert abs(dauer - 50 * anzahl / 30) < 0.15



def test_jede_faltstufe_traegt_feste_bildrate() -> None:
    """Der eigentliche Fix (rein, ohne ffmpeg): vor jedem settb=AVTB steht fps=<Rate>."""
    from laura.render.mp4 import _xfade_base_graph
    clips = [(Path(f"c{i}.mp4"), 0, 50) for i in range(4)]
    teile, _v, _a = _xfade_base_graph(clips, xdur=[6, 0, 6, 0], audio_flags=[True] * 4, has_base_audio=True,
                                      rate_num=30, rate_den=1)
    video = [t for t in teile if "settb=AVTB" in t]
    assert video and all("fps=30/1,settb=AVTB" in t for t in video)
