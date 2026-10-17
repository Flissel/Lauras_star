"""Parse ffprobe output into canonical asset metadata + a content hash."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ffmpeg import probe as ffprobe


@dataclass(frozen=True)
class AssetProbe:
    type: str                      # 'video' | 'audio'
    duration_frames: int | None
    rate_num: int | None
    rate_den: int | None
    audio_sample_rate: int | None
    start_timecode: str | None
    width: int | None
    height: int | None
    codec_video: str | None
    codec_audio: str | None
    is_vfr: bool


def _parse_rational(value: str | None) -> tuple[int, int] | None:
    if not value or "/" not in value:
        return None
    num_s, den_s = value.split("/", 1)
    try:
        num, den = int(num_s), int(den_s)
    except ValueError:
        return None
    if den == 0:
        return None
    return num, den


def _first(streams: list[dict[str, Any]], codec_type: str) -> dict[str, Any] | None:
    return next((s for s in streams if s.get("codec_type") == codec_type), None)


def _find_timecode(video: dict[str, Any] | None, fmt: dict[str, Any]) -> str | None:
    for tags in ((video or {}).get("tags", {}), fmt.get("tags", {})):
        tc = tags.get("timecode")
        if tc:
            return str(tc)
    return None


_BILD_FORMATE = frozenset({"image2", "png_pipe", "jpeg_pipe", "webp_pipe", "bmp_pipe"})
_BILD_CODECS = frozenset({"png", "mjpeg", "webp", "bmp"})


def _ist_bild(fmt: dict[str, Any], video: dict[str, Any] | None,
              audio: dict[str, Any] | None, duration_frames: int | None) -> bool:
    """Ein Bild: Bild-Container ODER ein Bild-Codec mit hoechstens einem Frame, ohne Ton.
    Ein Motion-JPEG-Video (mjpeg, viele Frames) bleibt ein Video."""
    if video is None or audio is not None:
        return False
    formate = set(str(fmt.get("format_name") or "").split(","))
    if formate & _BILD_FORMATE:
        return True
    return video.get("codec_name") in _BILD_CODECS and (duration_frames or 0) <= 1


def parse_probe(data: dict[str, Any]) -> AssetProbe:
    streams = data.get("streams", [])
    fmt = data.get("format", {})
    video = _first(streams, "video")
    audio = _first(streams, "audio")

    rate = _parse_rational(video.get("r_frame_rate")) if video else None
    avg = _parse_rational(video.get("avg_frame_rate")) if video else None
    rate_num, rate_den = rate if rate else (None, None)

    # VFR heuristic: real (r_frame_rate) and average rate disagree.
    is_vfr = bool(rate and avg and rate != avg and avg != (0, 1))

    duration_frames: int | None = None
    if video is not None:
        nb = video.get("nb_frames")
        if isinstance(nb, (str, int)):
            try:
                duration_frames = int(nb)
            except ValueError:
                duration_frames = None
        if duration_frames is None and rate:
            dur = fmt.get("duration") or video.get("duration")
            if isinstance(dur, (str, int, float)):
                duration_frames = round(float(dur) * rate[0] / rate[1])

    sample_rate: int | None = None
    if audio is not None and audio.get("sample_rate") not in (None, "N/A"):
        try:
            sample_rate = int(audio["sample_rate"])
        except ValueError:
            sample_rate = None

    if _ist_bild(fmt, video, audio, duration_frames):
        # Standbild (Video-Claw, 17.10.2026): eigener Typ, damit es nicht als
        # Ein-Frame-"Video" in Proxy/Analyse faellt; Clips daraus macht baustein.standbild.
        return AssetProbe(
            type="image", duration_frames=None, rate_num=None, rate_den=None,
            audio_sample_rate=None, start_timecode=None,
            width=int(video["width"]) if video and "width" in video else None,
            height=int(video["height"]) if video and "height" in video else None,
            codec_video=video.get("codec_name") if video else None, codec_audio=None,
            is_vfr=False,
        )

    return AssetProbe(
        type="video" if video is not None else "audio",
        duration_frames=duration_frames,
        rate_num=rate_num,
        rate_den=rate_den,
        audio_sample_rate=sample_rate,
        start_timecode=_find_timecode(video, fmt),
        width=int(video["width"]) if video and "width" in video else None,
        height=int(video["height"]) if video and "height" in video else None,
        codec_video=video.get("codec_name") if video else None,
        codec_audio=audio.get("codec_name") if audio else None,
        is_vfr=is_vfr,
    )


def probe_asset(path: Path | str) -> AssetProbe:
    return parse_probe(ffprobe(path))


def sha256_file(path: Path | str, *, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (used for content-based relink — docs/06-storage.md)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()
