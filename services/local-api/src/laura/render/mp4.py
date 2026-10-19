"""Render a list of timeline clip segments into one MP4 via ffmpeg (trim + concat).

Hard cuts use a single ``concat`` (byte-identical to the pre-crossfade renderer). When any
boundary carries a real ``crossfade`` transition the video/audio assembly switches to a
pairwise fold: ``xfade``/``acrossfade`` with a **reserve overlap** (clip A is extended by the
transition duration into its post-cut source handles), so the assembled length stays exactly
the sum of clip content lengths (sync invariant) and B loses no content.

Multi-lane compositing (§7 of the free-multitrack spec):
When ``render_multilane_mp4`` is called with clips on multiple lanes, each lane is assembled
as an independent segment chain (real clips interleaved with colour/transparent gap segments),
then lanes are stacked via ``overlay=eof_action=pass`` in ascending lane order (lane 0 is the
opaque base; lanes ≥ 1 carry ``yuva420p`` transparent gaps so lower lanes show through).
Single-lane timelines delegate to ``render_clips_mp4`` byte-identically.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..ingest.ffmpeg import FFmpegError, probe, run_ffmpeg
from .audio import AudioOverlay
from .reel import _drawtext_pfad, reel_blur_fill_graph, reel_video_chain, resolve_font
from .zoom import ZoomSpec, zoom_concat_graph

logger = logging.getLogger(__name__)

_DEFAULT_DISCLOSURE: str = "KI · synthetisch"

# ffmpeg takes the filtergraph on the command line, and Windows caps a command line near
# 32k chars. A long cut — many segments, each with its own zoom/crop chain — blows past
# that: CreateProcess fails with WinError 206, which run_ffmpeg reports as the misleading
# "ffmpeg not found". Graphs above this budget are handed to ffmpeg as a file instead
# (-filter_complex_script). Live finding: a 55-segment multi-window cut died where the
# same pipeline rendered 13 segments fine. The budget stays well under the cap because
# the inputs and encoder flags share the same command line.
_MAX_INLINE_FILTER_CHARS: int = 8000


def _mixed_size_normalize(clips: list[tuple[Path, int, int]]) -> str | None:
    """A per-clip ``scale``/``pad``/``setsar`` chain when the sources disagree on frame size.

    ffmpeg's ``concat`` (and ``xfade``) demand identical frame parameters on every input, and
    screen recordings practically never agree — each capture sits a few pixels of window
    chrome apart. Live 2026-08-03: the first real multi-source overview render (1916x1030 +
    1920x1026) died with "Input link parameters do not match"; every earlier live run happened
    to draw all clips from ONE source, so the mixed path had never executed.

    Target canvas = the max width x max height over the sources (rounded up to even for
    yuv420p); every clip is fitted without cropping and padded centered, SAR pinned to 1.
    Uniform sources return ``None`` so those paths stay byte-identical to today. A source
    whose probe fails contributes nothing (the render will surface that error itself).
    """
    sizes: dict[Path, tuple[int, int]] = {}
    for src, _fin, _fout in clips:
        if src in sizes:
            continue
        try:
            streams = probe(src).get("streams", [])
            video: dict[str, object] = next(
                (s for s in streams if s.get("codec_type") == "video"), {}
            )
            sizes[src] = (int(str(video.get("width", 0))), int(str(video.get("height", 0))))
        except (FFmpegError, OSError, ValueError, TypeError):
            continue
    unique = {s for s in sizes.values() if s[0] > 0 and s[1] > 0}
    if len(unique) <= 1:
        return None
    width = max(s[0] for s in unique)
    height = max(s[1] for s in unique)
    width += width % 2
    height += height % 2
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    )


def _effective_disclosure(disclosure_text: str | None) -> str | None:
    """Map disclosure_text to the effective value for the renderer.

    - ``None``        → ``None``  (no overlay; plain export, not a reel)
    - ``""`` / blank  → ``_DEFAULT_DISCLOSURE``  (explicit blank cannot suppress a
                        requested disclosure — EU AI Act enforcement for the reel path)
    - non-blank str   → stripped text
    """
    if disclosure_text is None:
        return None
    text = disclosure_text.strip()
    return text if text else _DEFAULT_DISCLOSURE


@dataclass(frozen=True)
class VideoTransition:
    kind: str
    boundary_frame: int
    duration_frames: int


def _seconds(value: float) -> str:
    if abs(value) < 0.000001:
        return "0"
    rounded = round(value, 6)
    if float(rounded).is_integer():
        return str(int(rounded))
    return str(rounded)


def _music_overlays(
    music_tracks: list[tuple[Path, int, int, int]] | None,
) -> list[AudioOverlay]:
    return [
        AudioOverlay(
            path=path,
            seq_in_frame=seq_in,
            seq_out_frame_exclusive=seq_out,
            gain_percent=gain_percent,
        )
        for path, seq_in, seq_out, gain_percent in (music_tracks or [])
    ]


def _source_has_audio(path: Path) -> bool:
    try:
        data = probe(path)
    except FFmpegError:
        return False
    streams = data.get("streams", [])
    if not isinstance(streams, list):
        return False
    return any(
        isinstance(stream, dict) and stream.get("codec_type") == "audio"
        for stream in streams
    )


def _video_transition_chain(
    video_transitions: list[VideoTransition] | None,
    *,
    rate_num: int,
    rate_den: int,
    total_frames: int,
) -> str:
    """Build the dip-to-black filter chain for the concat/xfade video streams.

    ``total_frames`` is the assembled stream's own total length (frames). It exists to guard
    against a ffmpeg ``fade`` filter trap, live-found on a narrated-reel trailing fade
    (I-1 follow-up) and confirmed independently on a MID-timeline dip (I-2): ``fade=t=in:st=X``
    does NOT merely ramp up during ``[X, X+d]`` and leave everything before ``X`` untouched --
    it forces EVERY frame before ``X`` to black, no matter how far before. Symmetrically,
    ``fade=t=out:st=X:d=D`` does not merely ramp down during ``[X, X+D]`` and leave the frames
    after that window untouched -- it STAYS black for the whole remainder after its own window.
    Empirically confirmed both ways with a solid-white clip + ``signalstats`` YAVG. Chained
    unguarded over one stream (fade-out then fade-in around a boundary) the two halves therefore
    black EVERYTHING: fade-out blacks from its start to the stream's end, fade-in blacks from
    the stream's start to its own start -- together covering the whole timeline, not just the
    intended dip window around the boundary.

    The fix: each half carries ``enable='between(t,A,B)'`` (ffmpeg's per-filter timeline
    gating) so it only applies inside its own window ``[A, B]``; outside that window the
    filter is bypassed and the frame passes through unmodified (still bright). ``fade``'s
    st/d-based alpha is a pure function of the frame's own timestamp (not a running counter),
    so gating it with ``enable`` does not disturb the ramp shape inside the window -- verified
    via a real-ffmpeg lavfi probe (white source -> fade+enable -> signalstats) before relying on
    it here: bright before the window, dark at the boundary, bright again after.

    For a MID-timeline dip (boundary strictly before the stream's end) both halves are wanted --
    there IS real video after the boundary for the fade-in to ramp back up on, and the preceding
    fade-out already carries the video down to black right up to the same point. But a TRAILING
    fade (``boundary_frame >= total_frames``, spec §6 "letzter Clip Fade-out") has no frames at
    or after the boundary at all -- emitting its fade-in half doesn't ramp anything back up, it
    just force-blacks the stream that precedes it. So a trailing transition emits ONLY the
    fade-out half (its ``enable`` window still ends at ``boundary``, but since the stream has no
    frames past that point the gating is a no-op there -- verified this still stays black through
    the end, matching the pre-existing trailing-fade tests).
    """
    if not video_transitions:
        return ""
    filters: list[str] = []
    for transition in video_transitions:
        if transition.kind == "hard" or transition.duration_frames <= 0:
            continue
        duration = transition.duration_frames * rate_den / rate_num
        boundary = transition.boundary_frame * rate_den / rate_num
        fade_out_start = max(0.0, boundary - duration)
        fade_out_end = _seconds(boundary)
        filters.append(
            f"fade=t=out:st={_seconds(fade_out_start)}:d={_seconds(duration)}:"
            f"enable='between(t,{_seconds(fade_out_start)},{fade_out_end})'"
        )
        if transition.boundary_frame >= total_frames:
            continue  # trailing fade: no frames past the boundary for a fade-in to ramp up
        fade_in_end = _seconds(boundary + duration)
        filters.append(
            f"fade=t=in:st={_seconds(boundary)}:d={_seconds(duration)}:"
            f"enable='between(t,{_seconds(boundary)},{fade_in_end})'"
        )
    return ",".join(filters)


def _source_video_frames(path: Path) -> int:
    """Total frame count of ``path``'s first video stream (0 when unknown).

    Used to decide whether a clip has post-cut handles for a crossfade reserve. Prefers the
    container's ``nb_frames``; falls back to ``duration * avg_frame_rate``. Returns 0 when neither
    is available so the caller degrades the crossfade to a hard cut rather than guessing."""
    try:
        data = probe(path)
    except FFmpegError:
        return 0
    for stream in data.get("streams", []):
        if not isinstance(stream, dict) or stream.get("codec_type") != "video":
            continue
        nb = stream.get("nb_frames")
        if nb not in (None, "N/A"):
            try:
                return int(str(nb))
            except (TypeError, ValueError):
                pass
        dur = stream.get("duration") or data.get("format", {}).get("duration")
        rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
        try:
            num, den = (int(x) for x in str(rate).split("/"))
            if dur is not None and den:
                return int(float(dur) * num / den)
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    return 0


def _crossfade_durations(
    clips: list[tuple[Path, int, int]],
    video_transitions: list[VideoTransition] | None,
    *,
    source_frames: object | None = None,
) -> list[int]:
    """Per-boundary crossfade duration in frames (``0`` = hard cut), length ``len(clips) - 1``.

    Only ``kind == "crossfade"`` boundaries produce a non-zero entry. Each is clamped to the
    reserve actually available in clip A's source (frames after its out-point); when no reserve
    exists the boundary degrades to a hard cut (logged). ``source_frames`` is the frame-count
    function (defaults to :func:`_source_video_frames`); inject a stub in unit tests.
    """
    n = len(clips)
    out = [0] * max(0, n - 1)
    if not video_transitions or n < 2:
        return out
    frames_of = _source_video_frames if source_frames is None else source_frames
    by_boundary = {
        t.boundary_frame: t.duration_frames
        for t in video_transitions
        if t.kind == "crossfade" and t.duration_frames > 0
    }
    cum = 0
    for i in range(n - 1):
        cum += clips[i][2] - clips[i][1]  # cumulative content length up to end of clip i
        want = by_boundary.get(cum)
        if not want:
            continue
        avail = frames_of(clips[i][0]) - clips[i][2]  # type: ignore[operator]
        eff = min(int(want), max(0, int(avail)))
        out[i] = eff
        if eff < want:
            logger.warning(
                "crossfade at boundary %s shortened %s->%s frames (reserve=%s)",
                cum, want, eff, avail,
            )
    return out


@dataclass(frozen=True)
class LaneSegment:
    """One segment in a per-lane chain: either a real clip or a gap filler.

    ``path`` is ``None`` for a gap (filled with a colour source at render time).
    ``src_in`` / ``src_out`` are source-space frame indices (end-exclusive), meaningful only
    when ``path`` is not ``None``.
    ``seq_duration`` is the length of this segment **in sequence frames** (always > 0).
    ``has_audio`` is set by the caller after probing (only relevant for real clip segments on
    lane 0 that contribute to the base audio mix).
    """

    path: Path | None
    src_in: int
    src_out: int
    seq_duration: int
    has_audio: bool = field(default=False)


def build_lane_segments(
    lane_clips: list[tuple[Path, int, int, int, int]],
    total_seq_frames: int,
) -> list[LaneSegment]:
    """Convert a list of per-lane clip tuples into an ordered LaneSegment list.

    ``lane_clips`` is ``[(path, src_in, src_out, seq_in, seq_out_excl), ...]`` for one lane,
    already sorted by ``seq_in``.  ``total_seq_frames`` is the full sequence length across all
    lanes (determines where a trailing gap ends).

    Gaps between clips (and before the first clip / after the last clip) become ``LaneSegment``
    instances with ``path=None``.  Pure — no IO.
    """
    segments: list[LaneSegment] = []
    playhead = 0
    for path, src_in, src_out, seq_in, seq_out in lane_clips:
        if seq_in > playhead:
            # gap before this clip
            segments.append(
                LaneSegment(
                    path=None,
                    src_in=0,
                    src_out=0,
                    seq_duration=seq_in - playhead,
                )
            )
        segments.append(
            LaneSegment(
                path=path,
                src_in=src_in,
                src_out=src_out,
                seq_duration=seq_out - seq_in,
            )
        )
        playhead = seq_out
    # trailing gap
    if playhead < total_seq_frames:
        segments.append(
            LaneSegment(
                path=None,
                src_in=0,
                src_out=0,
                seq_duration=total_seq_frames - playhead,
            )
        )
    return segments


def build_multilane_filtergraph(
    lanes: list[list[LaneSegment]],
    *,
    rate_num: int,
    rate_den: int,
    base_input_offset: int = 0,
) -> tuple[list[tuple[Path, int, int]], str, str]:
    """Build a multi-lane ffmpeg filter_complex string for overlay compositing.

    Strategy (per §7 of the free-multitrack spec):

    **Lane 0 (base)**: real clips plus explicit opaque-black gap segments, concatenated into
    one continuous stream that covers the full sequence length.  Lane 0 drives the timing of
    the whole overlay stack.  Gap colour sources use ``color=c=black`` (no size — ffmpeg
    defaults to ``320x240``; the overlay filter scales the overlay to the base resolution, so
    size mismatches between gaps and real clips on lane 0 must be avoided by using the same
    default).  In practice, lane-0 timelines should not have gaps (the per-lane packing policy
    keeps lane 0 contiguous); the gap path is a safety net.

    **Lanes ≥ 1 (overlay)**: **no explicit gap segments** — instead each real clip is trimmed
    and its PTS is shifted by ``seq_in_frame / TB`` so it appears at the correct absolute
    sequence position.  The ``overlay=eof_action=pass:repeatlast=0`` filter passes the base
    through unchanged for timestamps where the overlay has no frames (before the clip starts
    and after it ends).  Multiple clips on the same overlay lane are each applied as a
    separate overlay stage (fold: ``base ← overlay_clip1 ← overlay_clip2 …``).  This avoids
    any size-mismatch concat issue and is the standard ffmpeg pattern for sparse overlays.

    Returns ``(real_clips, filter_complex, out_label)`` where:
    - ``real_clips`` is the ordered list of ``(path, src_in, src_out)`` for ALL real clip
      segments across all lanes, in the order the ``-i`` inputs must appear.
    - ``filter_complex`` is the full filter string (pass to ``-filter_complex``).
    - ``out_label`` is ``"[vout]"`` (the final composited video label).

    IMPORTANT: pure function — no IO, fully unit-testable.

    ``base_input_offset``: start index of real-clip inputs in the global ``-i`` list (non-zero
    when the caller prepends other ``-i`` entries before the video clips).

    Single-lane MUST NOT be routed through this function — the caller routes single-lane
    timelines to ``render_clips_mp4`` unchanged (byte-identity invariant).
    """
    assert len(lanes) >= 2, "build_multilane_filtergraph requires at least 2 lanes"

    parts: list[str] = []
    real_clips: list[tuple[Path, int, int]] = []
    input_idx = base_input_offset

    # --- Lane 0: concat chain with explicit gap segments ---
    lane0_segs = lanes[0]
    lane0_seg_labels: list[str] = []
    for seg_idx, seg in enumerate(lane0_segs):
        seg_label = f"[l0s{seg_idx}]"
        if seg.path is None:
            # Opaque black gap — no size specified; ffmpeg defaults to 320x240.
            dur_s = _seconds(seg.seq_duration * rate_den / rate_num)
            parts.append(
                f"color=c=black:r={rate_num}/{rate_den},"
                f"trim=duration={dur_s},setpts=PTS-STARTPTS{seg_label}"
            )
        else:
            real_clips.append((seg.path, seg.src_in, seg.src_out))
            parts.append(
                f"[{input_idx}:v]trim=start_frame={seg.src_in}:end_frame={seg.src_out},"
                f"setpts=PTS-STARTPTS,settb=AVTB{seg_label}"
            )
            input_idx += 1
        lane0_seg_labels.append(seg_label)

    n0 = len(lane0_seg_labels)
    if n0 == 1:
        parts.append(f"{lane0_seg_labels[0]}null[lane0]")
    else:
        joined0 = "".join(lane0_seg_labels)
        parts.append(f"{joined0}concat=n={n0}:v=1:a=0[lane0]")

    # Collect all overlay clip segments across all overlay lanes, in order.
    # Each entry: (lane_idx, seq_in_frame, seg) so we can build the overlay fold.
    overlay_clips: list[tuple[int, LaneSegment]] = []
    for lane_idx in range(1, len(lanes)):
        seq_start = 0
        for seg in lanes[lane_idx]:
            if seg.path is not None:
                overlay_clips.append((seq_start, seg))
            seq_start += seg.seq_duration

    # Sort overlay clips by their absolute seq_in_frame so the overlay fold is time-ordered.
    overlay_clips.sort(key=lambda t: t[0])

    # --- Lanes ≥ 1: PTS-offset overlay approach (no explicit gap segments) ---
    # Each real overlay clip is trimmed, then its PTS is shifted by seq_in_frame/TB so it
    # appears at the correct absolute sequence position.  The overlay filter's
    # eof_action=pass:repeatlast=0 passes the base through when no overlay frame exists
    # (before the clip starts or after it ends).  Multiple overlay clips are folded one-by-one.
    acc_label = "[lane0]"
    for k, (seq_in_frame, seg) in enumerate(overlay_clips):
        assert seg.path is not None
        real_clips.append((seg.path, seg.src_in, seg.src_out))
        offset_s = _seconds(seq_in_frame * rate_den / rate_num)
        clip_label = f"[ov{k}]"
        parts.append(
            f"[{input_idx}:v]trim=start_frame={seg.src_in}:end_frame={seg.src_out},"
            f"setpts=PTS-STARTPTS+{offset_s}/TB,"
            f"format=yuva420p{clip_label}"
        )
        input_idx += 1
        # Intermediate labels use [ols{k}]; the final fold emits [vout] directly.
        out_label = f"[ols{k}]" if k < len(overlay_clips) - 1 else "[vout]"
        parts.append(
            f"{acc_label}{clip_label}"
            f"overlay=eof_action=pass:repeatlast=0,format=yuv420p{out_label}"
        )
        acc_label = out_label

    # Edge case: no real overlay clips (all overlay lanes empty).
    # Shouldn't happen when the caller validates lanes, but handle gracefully.
    if not overlay_clips:
        parts.append("[lane0]null[vout]")

    return real_clips, ";".join(parts), "[vout]"


def _xfade_base_graph(
    clips: list[tuple[Path, int, int]],
    *,
    xdur: list[int],
    audio_flags: list[bool],
    has_base_audio: bool,
    rate_num: int,
    rate_den: int,
) -> tuple[list[str], str, str | None]:
    """Build the trim + pairwise xfade/concat filter parts for the crossfade path.

    ``xdur[i]`` is the crossfade duration (frames) for the boundary after clip ``i`` (0 = hard).
    A crossfade clip is trimmed ``xdur[i]`` frames past its out-point (reserve overlap). Returns
    ``(parts, video_label, audio_label_or_None)``. Pure — no IO, fully unit-testable.

    Every video node carries a uniform ``settb=AVTB`` timebase. ``xfade`` rejects inputs whose
    timebases differ ("main timebase do not match the corresponding second input link xfade
    timebase"); in a fold that mixes ``concat`` (passes input tb 1/1000000) and ``xfade`` (emits
    its own fps tb) the accumulator and the next raw clip otherwise disagree and the whole
    filtergraph fails to configure (-22). Pinning each segment + each fold output to ``AVTB``
    makes every pairwise stage see matching timebases regardless of cut/crossfade ordering.

    Each ``settb`` is preceded by ``fps=<sequence rate>`` (17.10.2026): with ffmpeg 7.1 (the Linux
    VM) the frame rate was lost at the output of a fold stage, and the NEXT ``xfade`` aborted with
    "The inputs needs to be a constant frame rate; current rate of 1/0" - every render with three
    or more crossfaded clips failed there. ffmpeg 8 tolerated it, the mocked tests never saw it."""
    n = len(clips)
    rate = f"fps={rate_num}/{rate_den}"
    parts: list[str] = []
    lens = [fout - fin for _, fin, fout in clips]
    # xfade is as strict as concat about frame parameters — mixed-size sources must be
    # normalized onto one canvas here too (None for uniform sources: unchanged graph).
    normalize = _mixed_size_normalize(clips)
    norm_chain = f",{normalize}" if normalize else ""
    for i, (_src, fin, fout) in enumerate(clips):
        reserve = xdur[i] if i < n - 1 else 0
        end = fout + reserve
        parts.append(
            f"[{i}:v]trim=start_frame={fin}:end_frame={end},"
            f"setpts=PTS-STARTPTS,{rate},settb=AVTB{norm_chain}[v{i}]"
        )
        if has_base_audio:
            if audio_flags[i]:
                start = _seconds(fin * rate_den / rate_num)
                end_s = _seconds(end * rate_den / rate_num)
                parts.append(
                    f"[{i}:a]atrim=start={start}:end={end_s},asetpts=PTS-STARTPTS[ba{i}]"
                )
            else:
                dur = _seconds((end - fin) * rate_den / rate_num)
                parts.append(
                    "anullsrc=channel_layout=stereo:sample_rate=48000,"
                    f"atrim=duration={dur},asetpts=PTS-STARTPTS[ba{i}]"
                )
    v_acc = "v0"
    a_acc: str | None = "ba0" if has_base_audio else None
    content = lens[0]
    for i in range(1, n):
        d = xdur[i - 1]
        if d > 0:
            offset = _seconds(content * rate_den / rate_num)
            dd = _seconds(d * rate_den / rate_num)
            nxt_v = f"xv{i}"
            parts.append(
                f"[{v_acc}][v{i}]xfade=transition=fade:duration={dd}:offset={offset},"
                f"{rate},settb=AVTB[{nxt_v}]"
            )
            v_acc = nxt_v
            if has_base_audio:
                nxt_a = f"xa{i}"
                parts.append(f"[{a_acc}][ba{i}]acrossfade=d={dd}[{nxt_a}]")
                a_acc = nxt_a
        else:
            nxt_v = f"cv{i}"
            parts.append(f"[{v_acc}][v{i}]concat=n=2:v=1:a=0,{rate},settb=AVTB[{nxt_v}]")
            v_acc = nxt_v
            if has_base_audio:
                nxt_a = f"ca{i}"
                parts.append(f"[{a_acc}][ba{i}]concat=n=2:v=0:a=1[{nxt_a}]")
                a_acc = nxt_a
        content += lens[i]
    return parts, v_acc, a_acc


def untertitel_filter(ass_basename: str | None, srt_basename: str | None, fontsdir: str | None) -> str:
    """Untertitel-Filter; mit ``fontsdir`` findet libass die Markenschriften (Marke in Reels, C1)."""
    if ass_basename:
        if fontsdir:
            # Maskierung wie drawtext (Windows-Laufwerk C\\:, Schraegstriche); unter Linux unveraendert
            return f"ass={ass_basename}:fontsdir={_drawtext_pfad(fontsdir.replace(chr(92), '/'))}"
        return f"ass={ass_basename}"
    if srt_basename:
        return f"subtitles={srt_basename}"
    return ""


def render_clips_mp4(
    clips: list[tuple[Path, int, int]],
    dest: Path,
    *,
    rate_num: int,
    rate_den: int,
    music_tracks: list[tuple[Path, int, int, int]] | None = None,
    audio_overlays: list[AudioOverlay] | None = None,
    vertical: bool = False,
    reel_fit: bool = False,
    reel_blur_fill: bool = False,
    out_size: tuple[int, int] | None = None,
    hook_text: str | None = None,
    disclosure_text: str | None = None,
    caption_ass: str | None = None,
    caption_srt: str | None = None,
    video_transitions: list[VideoTransition] | None = None,
    loudnorm: bool = False,
    zoom_specs: list[ZoomSpec | None] | None = None,
    caption_fontsdir: str | None = None,
    hook_font: str | None = None,
) -> None:
    """Trim each clip by frame range (end-exclusive) and concat into one MP4.

    ``music_tracks`` is an optional list of
    ``(path, seq_in_frame, seq_out_frame_exclusive, gain_percent)`` tuples.
    Each track is mixed into the output audio at the specified position:

    * ``seq_in_frame`` / ``seq_out_frame_exclusive`` define the window inside
      the assembled sequence where the track should be audible (integer frames,
      end-exclusive — consistent with the frame-accuracy invariant).
    * ``gain_percent`` is applied as a ``volume`` filter (100 = unity gain).

    With no tracks (or ``music_tracks=None``) the output is video-only
    (unchanged — backward compatible).

    ``audio_overlays`` is the structured v2 equivalent for sequence-level audio
    clips. It adds ``asset_in_frame`` and simple fade-in/out frame ranges while
    preserving the same end-exclusive sequence placement.

    ``caption_ass`` is an optional complete ASS document string (as returned by
    ``build_ass``).  When set, the subtitles are burned into the output using
    the ffmpeg ``ass=`` filter.  The file is written next to *dest*, added to
    the ``reel_files`` cleanup list, and removed after the render regardless of
    success or failure.  Defaults to ``None`` (no captions, byte-identical to
    the pre-R1.2 behaviour).

    ``caption_srt`` is an optional complete SRT document string (as returned by
    ``sequence_transcript_to_srt``).  When set, the subtitles are burned into
    the output using the ffmpeg ``subtitles=`` filter.  The file is written next
    to *dest*, cleaned up afterwards, and the ``subtitles=`` filter is appended
    after any ASS caption filter.  Defaults to ``None`` (off).  When both
    ``caption_ass`` and ``caption_srt`` are provided, ASS takes precedence and
    the SRT is ignored.

    ``loudnorm`` (default ``False``) appends an EBU R128 ``loudnorm`` stage
    (``I=-14:TP=-1.5:LRA=11`` — the common social-media loudness target) as the
    final step of the audio graph when the output has audio.  With it off (or no
    audio) the audio graph is byte-identical to the pre-loudnorm behaviour.

    ``reel_fit`` (default ``False``) switches the vertical reframe from center-crop to
    letterbox fit when ``vertical=True``.  Use for screencasts or any content where
    center-crop would slice off readable text.  Has no effect when ``vertical=False``.

    ``reel_blur_fill`` (default ``False``) switches to *blurred-background fill* mode when
    ``vertical=True``.  The source frame is scaled to fit 1080×1920 (no cropping), and the
    top/bottom dead space is filled with a heavily blurred, scale-to-cover copy of the same
    frame.  This produces the Instagram/TikTok "reel fill" look instead of black bars.
    Precedence: ``reel_blur_fill`` > ``reel_fit`` > center-crop (default).  Has no effect
    when ``vertical=False`` or when ``vertical=True`` but ``reel_blur_fill=False``.
    Because the blur-fill requires a ``split``/``overlay`` sub-graph it is wired directly
    into the ``filter_complex`` string rather than through the comma-chain returned by
    :func:`reel_video_chain` — drawtext and ASS captions are still applied on top of the
    composited 1080×1920 stream, so caption output is identical to the other vertical modes.

    ``zoom_specs`` (default ``None``) switches on the ``zoom_hybrid`` fit mode: a
    ``list[ZoomSpec | None]`` that must align 1:1 with ``clips`` (``None`` entries fall back to
    plain blur-fill for that segment).  When set, the video graph is built by
    :func:`zoom_concat_graph` (per-segment blurred full frame dissolving into a static
    end-window crop) instead of the crossfade/concat assembly; captions, disclosure/hook
    drawtext, and the audio graph are layered on top exactly as in the other paths — only the
    ``[vcat]``-producing video branch differs.  v1 scope: requires ``vertical=True`` and excludes
    ``video_transitions`` (both raise ``ValueError``), and ``len(zoom_specs) != len(clips)``
    raises ``ValueError``.  With ``zoom_specs=None`` (default) every path is byte-identical to
    today.
    """
    if zoom_specs is not None:
        if len(zoom_specs) != len(clips):
            raise ValueError("zoom_specs must align 1:1 with clips")
        if not vertical:
            raise ValueError("zoom_hybrid requires vertical=True")
        if video_transitions:
            raise ValueError("zoom_hybrid excludes video transitions (v1)")
    use_zoom = zoom_specs is not None
    dest.parent.mkdir(parents=True, exist_ok=True)
    inputs: list[str] = []
    audio_flags = [_source_has_audio(src) for src, _, _ in clips]
    has_base_audio = any(audio_flags)
    # Total assembled length in frames -- the sync invariant (module docstring) holds this equal
    # to the sum of clip content lengths on BOTH the concat and xfade paths (a crossfade extends
    # clip A into its post-cut reserve rather than shortening the timeline). Used to detect a
    # TRAILING fade transition (boundary_frame >= this) so _video_transition_chain can skip the
    # poisonous fade-in half (see its docstring).
    total_frames = sum(fout - fin for _src, fin, fout in clips)
    for src, _fin, _fout in clips:
        inputs += ["-i", str(src)]
    # Per-boundary crossfade durations (frames; 0 = hard cut). Any non-zero entry switches the
    # video/audio assembly to the xfade/acrossfade fold; otherwise the byte-identical concat path
    # (which also renders the dip-to-black ``fade`` transition via _video_transition_chain).
    xdur = _crossfade_durations(clips, video_transitions)
    has_crossfade = any(d > 0 for d in xdur)
    # Transitions eligible for the dip-to-black chain (_video_transition_chain): "hard" cuts
    # never dip, and "crossfade" boundaries are either folded into the xfade graph above OR
    # degraded to a silent hard cut by _crossfade_durations when the outgoing clip has no
    # post-cut reserve -- either way they must NOT also reach _video_transition_chain, which
    # would emit a dip-to-black fade=t=out/fade=t=in pair for them. A degraded crossfade still
    # carries kind="crossfade" with its original (unclamped) duration, so filtering by kind here
    # (not by xdur) is what keeps a "shortened 8->0" boundary from becoming a dip-to-black.
    # Computed ONCE so the xfade and concat branches below share one value and cannot drift
    # apart on this filter again (live 2026-08-22: the concat branch had never gotten it).
    dip_transitions = [
        t for t in (video_transitions or []) if t.kind not in ("hard", "crossfade")
    ]

    # Reel overlay text goes through drawtext ``textfile=`` (basename, resolved via
    # ffmpeg's cwd), NOT inline ``text='…'`` — inline text cannot be escaped reliably
    # for arbitrary user input on Windows and is a filtergraph-injection vector.
    # Write the UTF-8 files next to dest, reference by basename, run ffmpeg with
    # cwd=dest.parent, and always clean the files up afterwards.
    reel_files: list[Path] = []
    hook_tf: str | None = None
    disc_tf: str | None = None
    if hook_text:
        hook_path = dest.parent / f"{dest.stem}.reel_hook.txt"
        hook_path.write_text(hook_text, encoding="utf-8")
        reel_files.append(hook_path)
        hook_tf = hook_path.name
    disclosure = _effective_disclosure(disclosure_text)
    if disclosure is not None:
        disc_path = dest.parent / f"{dest.stem}.reel_disclosure.txt"
        disc_path.write_text(disclosure, encoding="utf-8")
        reel_files.append(disc_path)
        disc_tf = disc_path.name

    ass_basename: str | None = None
    if caption_ass:
        ass_path = dest.parent / f"{dest.stem}.reel_caption.ass"
        ass_path.write_text(caption_ass, encoding="utf-8")
        reel_files.append(ass_path)
        ass_basename = ass_path.name

    # SRT burn-in: only used when no ASS captions are present (ASS takes precedence).
    srt_basename: str | None = None
    if caption_srt and not caption_ass:
        srt_path = dest.parent / f"{dest.stem}.export_caption.srt"
        srt_path.write_text(caption_srt, encoding="utf-8")
        reel_files.append(srt_path)
        srt_basename = srt_path.name

    try:
        # Determine which vertical reframe mode is active.  Precedence:
        #   reel_blur_fill (split/overlay) > reel_fit (letterbox) > center-crop (default).
        # reel_blur_fill requires a split/overlay sub-graph that cannot be expressed as a
        # simple comma chain, so it is wired directly into filter_complex (see below).
        # reel_video_chain is called without vertical=True when blur-fill handles reframing,
        # so it only emits the drawtext/caption chain without any crop/scale/pad filters.
        use_blur_fill = vertical and reel_blur_fill
        # Target canvas for any reframe mode; default is the classic 1080×1920 reel.
        out_w, out_h = out_size if out_size is not None else (1080, 1920)
        reel = reel_video_chain(
            vertical=vertical and not use_blur_fill and not use_zoom,
            reel_fit=reel_fit,
            hook_textfile=hook_tf,
            disclosure_textfile=disc_tf,
            font=resolve_font(),
            hook_font=_drawtext_pfad(hook_font.replace(chr(92), "/")) if hook_font else None,
            out_w=out_w,
            out_h=out_h,
        )
        caption_filter = untertitel_filter(ass_basename, srt_basename, caption_fontsdir)
        if use_zoom:
            # zoom_hybrid: per-segment blurred-full-frame -> static end-window crop dissolve
            # (Task 7's zoom_concat_graph). Emits "[abase]" directly when has_base_audio, which
            # is exactly the label the overlay-ducking/loudnorm code below expects — no extra
            # remap stage needed (unlike the xfade fold's [a_label] -> [abase] step below).
            assert zoom_specs is not None
            zparts, v_label, _a_label = zoom_concat_graph(
                [(fin, fout) for _src, fin, fout in clips],
                zoom_specs,
                audio_flags=audio_flags,
                has_base_audio=has_base_audio,
                rate_num=rate_num,
                rate_den=rate_den,
                out_w=out_w,
                out_h=out_h,
            )
            post_caption = ",".join(p for p in (reel, caption_filter) if p)
            if post_caption:
                parts = f"{zparts};{v_label}{post_caption}[out]"
            else:
                parts = f"{zparts};{v_label}null[out]"
        elif has_crossfade:
            # Real cross-dissolve: pairwise xfade/acrossfade fold with reserve overlap.
            fold_parts, v_label, a_label = _xfade_base_graph(
                clips,
                xdur=xdur,
                audio_flags=audio_flags,
                has_base_audio=has_base_audio,
                rate_num=rate_num,
                rate_den=rate_den,
            )
            # The xfade fold above only handles kind=="crossfade" boundaries (via `xdur`);
            # any other non-hard transition (e.g. a trailing "fade" on the final clip, spec
            # §6 "letzter Clip Fade-out") is NOT part of the fold and would otherwise be
            # silently dropped on this path (I-1 reader gap 2). Apply it here, mirroring the
            # concat path below: same dip-to-black chain, same position (before reel/captions).
            transition_chain = _video_transition_chain(
                dip_transitions, rate_num=rate_num, rate_den=rate_den, total_frames=total_frames
            )
            if use_blur_fill:
                # Insert blur-fill sub-graph between the xfade output and captions.
                # The sub-graph takes [v_label] → [_rbout]; captions are applied after.
                blur_in_label = f"[{v_label}]"
                if transition_chain:
                    fold_parts.append(f"[{v_label}]{transition_chain}[_xftrans]")
                    blur_in_label = "[_xftrans]"
                blur_out = "[_rbout]"
                blur_graph = reel_blur_fill_graph(blur_in_label, blur_out, out_w=out_w, out_h=out_h)
                fold_parts.append(blur_graph)
                post_caption = ",".join(p for p in (reel, caption_filter) if p)
                if post_caption:
                    fold_parts.append(f"{blur_out}{post_caption}[out]")
                else:
                    fold_parts.append(f"{blur_out}null[out]")
            else:
                post = ",".join(p for p in (transition_chain, reel, caption_filter) if p)
                fold_parts.append(
                    f"[{v_label}]{post}[out]" if post else f"[{v_label}]null[out]"
                )
            if has_base_audio and a_label is not None:
                fold_parts.append(f"[{a_label}]anull[abase]")
            parts = ";".join(fold_parts)
        else:
            # Byte-identical concat path (hard cuts + optional dip-to-black ``fade``) — except
            # when the sources disagree on frame size, where concat would refuse to configure
            # (see _mixed_size_normalize); uniform sources get an unchanged graph.
            normalize = _mixed_size_normalize(clips)
            norm_chain = f",{normalize}" if normalize else ""
            filt: list[str] = []
            for i, (_src, fin, fout) in enumerate(clips):
                # Frame-exact, end-exclusive: trim keeps input frames [start_frame, end_frame).
                filt.append(
                    f"[{i}:v]trim=start_frame={fin}:end_frame={fout},"
                    f"setpts=PTS-STARTPTS{norm_chain}[v{i}]"
                )
                if has_base_audio:
                    start = fin * rate_den / rate_num
                    end = fout * rate_den / rate_num
                    duration = (fout - fin) * rate_den / rate_num
                    if audio_flags[i]:
                        filt.append(
                            f"[{i}:a]atrim=start={_seconds(start)}:end={_seconds(end)},"
                            f"asetpts=PTS-STARTPTS[ba{i}]"
                        )
                    else:
                        filt.append(
                            "anullsrc=channel_layout=stereo:sample_rate=48000,"
                            f"atrim=duration={_seconds(duration)},asetpts=PTS-STARTPTS[ba{i}]"
                        )
            concat_in = (
                "".join(f"[v{i}][ba{i}]" for i in range(len(clips)))
                if has_base_audio
                else "".join(f"[v{i}]" for i in range(len(clips)))
            )
            transition_chain = _video_transition_chain(
                dip_transitions, rate_num=rate_num, rate_den=rate_den, total_frames=total_frames
            )
            if use_blur_fill:
                # Blur-fill path: concat output goes to [vcat], blur-fill sub-graph takes it
                # to [_rbout], then caption filters (drawtext/ass) are applied after.
                # transition_chain still applies before the concat output if present.
                pre_blur = ",".join(p for p in (transition_chain,) if p)
                concat_out = "[vcat]"
                if has_base_audio:
                    parts = (
                        ";".join(filt)
                        + f";{concat_in}concat=n={len(clips)}:v=1:a=1{concat_out}[abase]"
                    )
                else:
                    parts = (
                        ";".join(filt)
                        + f";{concat_in}concat=n={len(clips)}:v=1:a=0{concat_out}"
                    )
                # Apply transition chain (e.g. fade) to vcat before blur-fill if any.
                blur_in_label = "[vcat]"
                if pre_blur:
                    parts += f";[vcat]{pre_blur}[_rbtrans]"
                    blur_in_label = "[_rbtrans]"
                blur_graph = reel_blur_fill_graph(
                    blur_in_label, "[_rbout]", out_w=out_w, out_h=out_h
                )
                parts += f";{blur_graph}"
                # Drawtext and ASS captions are applied on the composited 1080×1920 stream.
                post_caption = ",".join(p for p in (reel, caption_filter) if p)
                if post_caption:
                    parts += f";[_rbout]{post_caption}[out]"
                else:
                    parts += ";[_rbout]null[out]"
            else:
                post = ",".join(p for p in (transition_chain, reel, caption_filter) if p)
                concat_out = "[vcat]" if post else "[out]"
                if has_base_audio:
                    parts = (
                        ";".join(filt)
                        + f";{concat_in}concat=n={len(clips)}:v=1:a=1{concat_out}[abase]"
                    )
                else:
                    parts = (
                        ";".join(filt)
                        + f";{concat_in}concat=n={len(clips)}:v=1:a=0{concat_out}"
                    )
                if post:
                    parts += f";[vcat]{post}[out]"

        audio_maps: list[str] = []
        overlays = [*_music_overlays(music_tracks), *(audio_overlays or [])]
        audio_labels: list[str] = []
        if has_base_audio:
            base_label = "[abase]"
            for i, overlay in enumerate(overlays):
                ducking = (
                    0
                    if overlay.mix_mode in {"replace_original", "mute_original"}
                    else overlay.ducking_percent
                )
                if ducking >= 100:
                    continue
                next_label = f"[aduck{i}]"
                duck_start = _seconds(overlay.seq_in_frame * rate_den / rate_num)
                duck_end = _seconds(overlay.seq_out_frame_exclusive * rate_den / rate_num)
                parts += (
                    f";{base_label}volume=enable='between(t,{duck_start},{duck_end})'"
                    f":volume={ducking / 100}{next_label}"
                )
                base_label = next_label
            audio_labels.append(base_label)

        if overlays:
            n_vid = len(clips)
            for i, overlay in enumerate(overlays):
                duration_frames = overlay.seq_out_frame_exclusive - overlay.seq_in_frame
                if duration_frames <= 0:
                    continue
                idx = n_vid + i
                inputs += ["-i", str(overlay.path)]
                asset_start = overlay.asset_in_frame * rate_den / rate_num
                duration = duration_frames * rate_den / rate_num
                start_ms = overlay.seq_in_frame * rate_den / rate_num * 1000
                fade_in = overlay.fade_in_frames * rate_den / rate_num
                fade_out = overlay.fade_out_frames * rate_den / rate_num
                # Build the per-track filter chain:
                #   volume  — scale the gain
                #   atrim   — cut the track to at most the scene window duration
                #   afade   — optional simple clip fades
                #   adelay  — push it to the right start position in the mix
                #   asetpts — reset PTS after the trim so adelay gets a clean base
                label = f"[a{i}]"
                chain = [
                    f"volume={overlay.gain_percent / 100}",
                    f"atrim=start={_seconds(asset_start)}:duration={_seconds(duration)}",
                    "asetpts=PTS-STARTPTS",
                ]
                if fade_in > 0:
                    chain.append(f"afade=t=in:st=0:d={_seconds(fade_in)}")
                if fade_out > 0:
                    fade_start = max(0.0, duration - fade_out)
                    chain.append(
                        f"afade=t=out:st={_seconds(fade_start)}:d={_seconds(fade_out)}"
                    )
                chain.append(f"adelay={start_ms:.6f}|{start_ms:.6f}")
                parts += f";[{idx}:a]" + ",".join(chain) + label
                audio_labels.append(label)

        # Assemble the final audio graph. With loudnorm on, the mixed/single track is folded
        # one extra stage through an EBU R128 loudnorm filter (social standard -14 LUFS) before
        # the final map; with it off this stage is skipped.
        apply_loudnorm = loudnorm and bool(audio_labels)
        # The penultimate label feeds the mandatory format-pin stage below. loudnorm (when on)
        # writes into it; otherwise the single-track anull / multi-track amix does.
        format_in = "[aout_pre]"
        if len(audio_labels) == 1:
            # Single track — route to the format-pin input (no amix needed).
            parts += f";{audio_labels[0]}anull{format_in}"
        elif len(audio_labels) > 1:
            joined = "".join(audio_labels)
            parts += f";{joined}amix=inputs={len(audio_labels)}:normalize=0{format_in}"

        if apply_loudnorm:
            # EBU R128 two-pass-style single-pass loudnorm; I/TP/LRA are the common social target.
            # loudnorm resamples internally (often to 192k/96k), so it MUST feed the pin below.
            parts += ";[aout_pre]loudnorm=I=-14:TP=-1.5:LRA=11[aout_loud]"
            format_in = "[aout_loud]"

        if audio_labels:
            # Pin the FINAL audio stream to one encoder-safe format at the single sink, regardless
            # of source rate (44.1k/48k/96k from loudnorm) or path (single clip, amix, acrossfade).
            # acrossfade/loudnorm/mixed sources otherwise leave an inconsistent rate that the AAC
            # encoder at fixed params rejects (-22 Invalid argument). aresample 48k->48k is a no-op.
            parts += f";{format_in}aresample=48000,aformat=channel_layouts=stereo[aout]"
            audio_maps = ["-map", "[aout]", "-c:a", "aac", "-ar", "48000", "-ac", "2"]

        # cwd is only needed so drawtext textfile= basenames resolve. Decide it BEFORE the
        # filtergraph file below joins reel_files, so the plain concat call stays
        # byte-identical (and any run_ffmpeg monkeypatch without a cwd kwarg keeps working).
        needs_cwd = bool(reel_files)
        if len(parts) > _MAX_INLINE_FILTER_CHARS:
            graph_path = dest.parent / f"{dest.stem}.filtergraph.txt"
            graph_path.write_text(parts, encoding="utf-8")
            reel_files.append(graph_path)  # cleaned up in the finally below
            filter_args = ["-filter_complex_script", str(graph_path)]
        else:
            filter_args = ["-filter_complex", parts]

        ff_args = [
            *inputs,
            *filter_args,
            "-map", "[out]",
            *audio_maps,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-r", f"{rate_num}/{rate_den}",
            str(dest),
        ]
        if needs_cwd:
            run_ffmpeg(ff_args, cwd=dest.parent)
        else:
            run_ffmpeg(ff_args)
    finally:
        for reel_file in reel_files:
            reel_file.unlink(missing_ok=True)


def render_multilane_mp4(
    lane_clip_rows: list[list[tuple[Path, int, int, int, int]]],
    dest: Path,
    *,
    rate_num: int,
    rate_den: int,
    music_tracks: list[tuple[Path, int, int, int]] | None = None,
    audio_overlays: list[AudioOverlay] | None = None,
    video_transitions: list[VideoTransition] | None = None,
    loudnorm: bool = False,
) -> None:
    """Render a multi-lane timeline to MP4 using an overlay compositing filter graph.

    ``lane_clip_rows`` is a list-of-lists indexed by lane number.  Each inner list contains
    ``(path, src_in, src_out, seq_in, seq_out_excl)`` tuples for the clips on that lane,
    already sorted by ``seq_in``.  The outer list must have at least 2 entries (for lane 0 and
    at least one overlay lane); lanes with no clips should be represented as empty lists and are
    skipped by the compositor.

    **Single-lane regression:** this function is NEVER called when only lane 0 has clips — the
    caller (``handlers.py``) routes single-lane timelines to ``render_clips_mp4`` unchanged to
    preserve byte-identity.

    Lane 0 is the opaque base chain; lanes ≥ 1 are transparent overlay chains stacked on top in
    ascending lane order (lane 1 directly above lane 0, lane 2 above lane 1, etc.).  Gaps (time
    spans on a lane with no clip) are filled with colour sources: opaque black for lane 0,
    transparent black (``alpha=0``) for overlay lanes so lower lanes show through.

    Audio comes only from lane-0 clips (V1) — overlay clips are silent B-roll in v1.  Music and
    ``audio_overlays`` work exactly as in ``render_clips_mp4``.  Captions and reel features are
    NOT exposed here (v1 scope: composite the video stack; captions/reels remain single-lane).
    """
    # Compute total sequence length = max seq_out across ALL lanes.
    total_seq = max(
        (seq_out for lane in lane_clip_rows for _, _, _, _, seq_out in lane),
        default=0,
    )
    if total_seq <= 0:
        raise ValueError("multi-lane render: total sequence length is 0")

    # Drop empty lanes (keep the lane index order; only lanes with clips participate in overlay).
    populated_lanes = [(idx, clips) for idx, clips in enumerate(lane_clip_rows) if clips]
    if not populated_lanes:
        raise ValueError("multi-lane render: no clips in any lane")

    # Build LaneSegment lists for each populated lane.
    lane_segments_list: list[list[LaneSegment]] = []
    for _lane_idx, clips in populated_lanes:
        segs = build_lane_segments(clips, total_seq)
        lane_segments_list.append(segs)

    # Gather real-clip paths for input ordering, detect base audio.
    # Real clips across all lanes in the order build_multilane_filtergraph will reference them.
    # We pre-scan to build the -i list in the same order.
    lane0_clips_for_audio: list[tuple[Path, int, int]] = [
        (p, si, so)
        for p, si, so, _, _ in (populated_lanes[0][1] if populated_lanes else [])
    ]
    audio_flags = [_source_has_audio(src) for src, _, _ in lane0_clips_for_audio]
    has_base_audio = any(audio_flags)

    # Build overlay inputs list first (music_tracks + audio_overlays) so we know the count
    # before the video inputs (which must come first to match filter indices).
    overlays = [*_music_overlays(music_tracks), *(audio_overlays or [])]

    # Build filter graph.  Real clip inputs start at index 0 in the filter (no audio-overlay
    # inputs come before video inputs — audio overlays are separate -i entries appended AFTER
    # the video inputs; their indices are computed after the video-input count is known).
    real_clips, filtergraph, vout_label = build_multilane_filtergraph(
        lane_segments_list,
        rate_num=rate_num,
        rate_den=rate_den,
        base_input_offset=0,
    )

    # Video inputs: one per real clip segment, in the order they appear in the filtergraph.
    inputs: list[str] = []
    for path, _, _ in real_clips:
        inputs += ["-i", str(path)]

    n_video_inputs = len(real_clips)

    # Reel post-processing and captions are not wired here (v1 scope).
    # We must close the filter with [vout] -> [out] for the -map.
    parts = filtergraph + ";[vout]null[out]"

    # Audio: lane-0 clips only.  Rebuild per-segment audio chains using the same logic as
    # render_clips_mp4 but only for the lane-0 real-clip segments (in input-index order).
    # Lane-0 clips appear first in real_clips (build_multilane_filtergraph processes lanes
    # in order), so input indices for lane-0 segments start at 0.
    audio_labels: list[str] = []
    if has_base_audio:
        # Build audio trim chain for each lane-0 clip segment.
        # The lane-0 clip segments occupy input indices 0 .. lane0_real_count-1.
        lane0_input_map: list[int] = []
        input_cursor = 0
        for seg in lane_segments_list[0]:
            if seg.path is not None:
                lane0_input_map.append(input_cursor)
                input_cursor += 1
            # gap segments have no input
        # Advance cursor through other lanes (not used for audio but needed for correctness check).

        abase_parts: list[str] = []
        a_idx = 0
        for local_seg_idx, seg in enumerate(lane_segments_list[0]):
            if seg.path is None:
                # Gap: silent padding
                dur_s = _seconds(seg.seq_duration * rate_den / rate_num)
                abase_parts.append(
                    f"anullsrc=channel_layout=stereo:sample_rate=48000,"
                    f"atrim=duration={dur_s},asetpts=PTS-STARTPTS[aba{a_idx}]"
                )
                a_idx += 1
            else:
                input_i = lane0_input_map[sum(
                    1 for s in lane_segments_list[0][:local_seg_idx] if s.path is not None
                )]
                fin = seg.src_in
                fout = seg.src_out
                start_s = _seconds(fin * rate_den / rate_num)
                end_s = _seconds(fout * rate_den / rate_num)
                dur_s = _seconds((fout - fin) * rate_den / rate_num)
                if audio_flags[sum(
                    1 for s in lane_segments_list[0][:local_seg_idx] if s.path is not None
                )]:
                    abase_parts.append(
                        f"[{input_i}:a]atrim=start={start_s}:end={end_s},"
                        f"asetpts=PTS-STARTPTS[aba{a_idx}]"
                    )
                else:
                    abase_parts.append(
                        f"anullsrc=channel_layout=stereo:sample_rate=48000,"
                        f"atrim=duration={dur_s},asetpts=PTS-STARTPTS[aba{a_idx}]"
                    )
                a_idx += 1

        # Concat all audio segments for lane 0.
        n_asegs = a_idx
        if n_asegs == 1:
            parts += ";[aba0]anull[abase]"
        else:
            joined_a = "".join(f"[aba{k}]" for k in range(n_asegs))
            parts += f";{joined_a}concat=n={n_asegs}:v=0:a=1[abase]"
        for seg_part in abase_parts:
            parts += f";{seg_part}"

        base_label = "[abase]"
        for i, overlay in enumerate(overlays):
            ducking = (
                0
                if overlay.mix_mode in {"replace_original", "mute_original"}
                else overlay.ducking_percent
            )
            if ducking >= 100:
                continue
            next_label = f"[aduck{i}]"
            duck_start = _seconds(overlay.seq_in_frame * rate_den / rate_num)
            duck_end = _seconds(overlay.seq_out_frame_exclusive * rate_den / rate_num)
            parts += (
                f";{base_label}volume=enable='between(t,{duck_start},{duck_end})'"
                f":volume={ducking / 100}{next_label}"
            )
            base_label = next_label
        audio_labels.append(base_label)

    # Audio overlay inputs (music, audio clips) — appended after video inputs.
    if overlays:
        for i, overlay in enumerate(overlays):
            duration_frames = overlay.seq_out_frame_exclusive - overlay.seq_in_frame
            if duration_frames <= 0:
                continue
            idx = n_video_inputs + i
            inputs += ["-i", str(overlay.path)]
            asset_start = overlay.asset_in_frame * rate_den / rate_num
            duration = duration_frames * rate_den / rate_num
            start_ms = overlay.seq_in_frame * rate_den / rate_num * 1000
            fade_in = overlay.fade_in_frames * rate_den / rate_num
            fade_out = overlay.fade_out_frames * rate_den / rate_num
            label = f"[a{i}]"
            chain = [
                f"volume={overlay.gain_percent / 100}",
                f"atrim=start={_seconds(asset_start)}:duration={_seconds(duration)}",
                "asetpts=PTS-STARTPTS",
            ]
            if fade_in > 0:
                chain.append(f"afade=t=in:st=0:d={_seconds(fade_in)}")
            if fade_out > 0:
                fade_start = max(0.0, duration - fade_out)
                chain.append(f"afade=t=out:st={_seconds(fade_start)}:d={_seconds(fade_out)}")
            chain.append(f"adelay={start_ms:.6f}|{start_ms:.6f}")
            parts += f";[{idx}:a]" + ",".join(chain) + label
            audio_labels.append(label)

    # Final audio mix.
    apply_loudnorm = loudnorm and bool(audio_labels)
    format_in = "[aout_pre]"
    if len(audio_labels) == 1:
        parts += f";{audio_labels[0]}anull{format_in}"
    elif len(audio_labels) > 1:
        joined_al = "".join(audio_labels)
        parts += f";{joined_al}amix=inputs={len(audio_labels)}:normalize=0{format_in}"

    audio_maps: list[str] = []
    if apply_loudnorm:
        parts += ";[aout_pre]loudnorm=I=-14:TP=-1.5:LRA=11[aout_loud]"
        format_in = "[aout_loud]"

    if audio_labels:
        parts += f";{format_in}aresample=48000,aformat=channel_layouts=stereo[aout]"
        audio_maps = ["-map", "[aout]", "-c:a", "aac", "-ar", "48000", "-ac", "2"]

    dest.parent.mkdir(parents=True, exist_ok=True)
    ff_args = [
        *inputs,
        "-filter_complex", parts,
        "-map", "[out]",
        *audio_maps,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", f"{rate_num}/{rate_den}",
        str(dest),
    ]
    run_ffmpeg(ff_args)
