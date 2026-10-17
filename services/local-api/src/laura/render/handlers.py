"""Job handler: render a timeline's clips to an MP4 export."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from .. import audit
from ..db import repos
from ..db.database import Database
from ..editing.otio_sync import resolve_clip_rows
from ..jobs.runner import JobContext, JobHandler
from ..ledger import get_ledger_store
from ..sequences.music import sequence_music_tracks
from ..sequences.transcript import sequence_transcript_blocks
from ..storage import publish, store_from_env
from .audio import AudioOverlay
from .captions import build_ass, group_caption_lines
from .captions_source import timeline_caption_words, voiceover_caption_words
from .mp4 import _DEFAULT_DISCLOSURE, VideoTransition, render_clips_mp4, render_multilane_mp4
from . import groesse
from .reel import kennzeichnung_text
from .srt import sequence_transcript_to_srt
from .sync import assert_or_fix_media_sync

_log = logging.getLogger(__name__)


def record_short_run_result(
    db: Database,
    options: dict[str, Any],
    *,
    status: str,
    trace: dict[str, Any] | None,
) -> None:
    """Update the short_run identified by ``options["short_run_id"]`` with *status* and *trace*.

    This is a fire-and-forget helper: if *short_run_id* is absent from *options* the call is a
    silent no-op; if ``update_run`` finds no matching row it returns ``None`` (also a no-op).
    All exceptions are caught and logged so a ledger hiccup NEVER converts a successful render
    into a failure, and NEVER masks the original error on the failure path.
    """
    run_id = options.get("short_run_id")
    if not run_id:
        return
    try:
        trace_json = json.dumps(trace) if trace is not None else None
        get_ledger_store(db).update_run(
            str(run_id),
            status=status,
            trace_json=trace_json,
        )
    except Exception:  # noqa: BLE001
        _log.exception("ledger update_run failed for run_id=%s (ignored)", run_id)

_CAPTION_PRESETS: dict[str, tuple[int, int]] = {
    "reels": (1080, 1920),
    "tiktok": (1080, 1920),
    "shorts": (1080, 1920),
    "wide": (1920, 1080),
}


def cap_clips_to_frames(
    clips: list[tuple[Path, int, int]], budget_frames: int
) -> list[tuple[Path, int, int]]:
    """Keep only the first ``budget_frames`` output frames of a ``(path, src_in, src_out)`` clip
    list (deterministic tail-trim for the reel duration cap): full clips until the budget, the
    boundary clip trimmed to fit, the rest dropped. Integer frames, end-exclusive. No-op when
    ``budget_frames <= 0`` or the clips already fit within the budget."""
    if budget_frames <= 0:
        return clips
    out: list[tuple[Path, int, int]] = []
    used = 0
    for path, fin, fout in clips:
        length = fout - fin
        if used + length <= budget_frames:
            out.append((path, fin, fout))
            used += length
            if used == budget_frames:
                break
        else:
            remaining = budget_frames - used
            if remaining > 0:
                out.append((path, fin, fin + remaining))
            break
    return out


def snap_budget_to_word_boundary(
    budget_frames: int, total_frames: int, word_seq_ends: list[int]
) -> int:
    """Snap a reel duration-cap budget *down* to the latest transcript word boundary at or below
    ``budget_frames``, so a capped reel ends on a complete word instead of mid-word.

    ``word_seq_ends`` are end-exclusive *sequence* frames (e.g. the third element of each
    ``timeline_caption_words`` token). Prefix-preserving: the result is only ever <= the budget,
    so the kept clips still start at seq 0 and burned captions / music stay aligned.

    Returns ``budget_frames`` unchanged whenever a snap would be a no-op: a non-positive budget,
    a cut that already fits (``total_frames <= budget_frames``), or no word boundary lying in
    ``(0, budget_frames]`` (e.g. B-roll with no transcript) — so the no-transcript path is identical
    to the plain ``cap_clips_to_frames`` tail-trim."""
    if budget_frames <= 0 or total_frames <= budget_frames:
        return budget_frames
    candidates = [w for w in word_seq_ends if 0 < w <= budget_frames]
    if not candidates:
        return budget_frames
    return max(candidates)


def _option_str(opts: dict[str, object], key: str, default: str, allowed: set[str]) -> str:
    value = opts.get(key)
    if not isinstance(value, str):
        return default
    return value if value in allowed else default


def _option_int(opts: dict[str, object], key: str, default: int, *, lo: int, hi: int) -> int:
    value = opts.get(key)
    if not isinstance(value, int):
        return default
    return max(lo, min(hi, value))


def _timeline_audio_overlays(db: Any, timeline_id: str) -> list[AudioOverlay]:
    overlays: list[AudioOverlay] = []
    for clip in repos.list_timeline_audio_clips(db, timeline_id):
        asset = repos.get_asset(db, clip["asset_id"])
        if asset is None:
            raise ValueError(f"audio asset not found: {clip['asset_id']}")
        overlays.append(
            AudioOverlay(
                path=Path(asset["source_path"]),
                seq_in_frame=int(clip["seq_in_frame"]),
                seq_out_frame_exclusive=int(clip["seq_out_frame_exclusive"]),
                asset_in_frame=int(clip["asset_in_frame"]),
                gain_percent=int(clip["gain_percent"]),
                fade_in_frames=int(clip["fade_in_frames"]),
                fade_out_frames=int(clip["fade_out_frames"]),
                mix_mode=str(clip.get("mix_mode") or "mix"),
                ducking_percent=int(clip.get("ducking_percent") or 100),
            )
        )
    return overlays


def _scene_length(db: Any, scene: dict[str, Any]) -> int:
    scene_timeline_id = scene.get("scene_timeline_id")
    if not scene_timeline_id:
        return 0
    clips = repos.list_timeline_clips(db, str(scene_timeline_id))
    return max((int(c["seq_out_frame_exclusive"]) for c in clips), default=0)


def _sequence_video_transitions(db: Any, sequence_timeline_id: str) -> list[VideoTransition]:
    transitions: list[VideoTransition] = []
    boundary = 0
    items = repos.list_sequence_items(db, sequence_timeline_id)
    for index, item in enumerate(items):
        scene = repos.get_scene(db, item["scene_id"])
        if scene is None:
            continue
        boundary += _scene_length(db, scene)
        if index >= len(items) - 1:
            continue
        kind = str(item.get("transition_after_kind") or "hard")
        duration_frames = int(item.get("transition_after_frames") or 0)
        if kind == "hard" or duration_frames <= 0:
            continue
        transitions.append(
            VideoTransition(
                kind=kind,
                boundary_frame=boundary,
                duration_frames=duration_frames,
            )
        )
    return transitions


def _clip_video_transitions(db: Any, timeline_id: str) -> list[VideoTransition]:
    """VideoTransitions for a rough_cut/scene timeline, from clip-level transition fields.

    Mirrors :func:`_sequence_video_transitions` but walks lane-0 clips: the boundary frame is a
    clip's ``seq_out_frame_exclusive`` (== the next clip's seq-in), and the transition that plays
    after it comes from ``transition_after_kind/frames``. The final clip has no following clip to
    fold a transition into, so a ``crossfade`` (or ``hard``) set on it is ignored -- but a trailing
    ``fade`` is still meaningful (a fade-out at the very end of the assembled video, spec §6
    "letzter Clip Fade-out") and IS emitted, with ``boundary_frame`` == the last clip's own
    ``seq_out_frame_exclusive`` (== total assembled length)."""
    transitions: list[VideoTransition] = []
    clips = [c for c in repos.list_timeline_clips(db, timeline_id) if int(c.get("lane") or 0) == 0]
    clips.sort(key=lambda c: int(c["seq_in_frame"]))
    for index, clip in enumerate(clips):
        kind = str(clip.get("transition_after_kind") or "hard")
        duration_frames = int(clip.get("transition_after_frames") or 0)
        is_last = index >= len(clips) - 1
        if is_last:
            if kind != "fade" or duration_frames <= 0:
                continue
        elif kind == "hard" or duration_frames <= 0:
            continue
        transitions.append(
            VideoTransition(
                kind=kind,
                boundary_frame=int(clip["seq_out_frame_exclusive"]),
                duration_frames=duration_frames,
            )
        )
    return transitions


def _timeline_is_synthetic(
    db: Database,
    timeline_id: str,
    audio_overlays: list[AudioOverlay],
) -> bool:
    """Return True if the timeline contains any synthetic/replacement content.

    Checks:
    1. audio_overlays with replace_original or mute_original mix_mode (already fetched).
    2. A single DB query for lane>=1 role='replace' clips OR clips whose asset has synthetic=1.
    """
    for overlay in audio_overlays:
        if overlay.mix_mode in {"replace_original", "mute_original"}:
            return True
    with db.connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM timeline_clips tc "
            "LEFT JOIN media_assets a ON a.id = tc.asset_id "
            "WHERE tc.timeline_id = ? "
            "AND (tc.role = 'replace' OR a.synthetic = 1) "
            "LIMIT 1",
            (timeline_id,),
        ).fetchone()
    return row is not None


_GESICHT_EFFEKTE = frozenset({"reenact", "lipsync"})


def _synthetic_kinds(db: Database, timeline_id: str) -> set[str]:
    """Welche Arten KI stecken in der Timeline? (fuer ``kennzeichnung="auto"``)

    - ``stimme``: ein Tonstueck aus einem synthetischen Voiceover-Asset - egal mit welchem
      mix_mode (die alte Pruefung sah eine untergemischte KI-Stimme nicht).
    - ``gesicht``: ein Videoclip aus einem Reenact-/Lipsync-Asset.
    - ``synthetisch``: alles andere Kuenstliche - sonstige synthetische Assets, Ersatzclips
      (role='replace') und ersetzter/stummgeschalteter Originalton aus Nicht-KI-Quellen
      (wie bisher als synthetisch gewertet).
    """
    arten: set[str] = set()
    with db.connection() as conn:
        for rolle, synth, effekt in conn.execute(
            "SELECT tc.role, a.synthetic, a.ai_effect FROM timeline_clips tc "
            "LEFT JOIN media_assets a ON a.id = tc.asset_id WHERE tc.timeline_id = ?",
            (timeline_id,),
        ).fetchall():
            if synth:
                arten.add("gesicht" if effekt in _GESICHT_EFFEKTE else "synthetisch")
            elif rolle == "replace":
                arten.add("synthetisch")
        for mix, synth, effekt in conn.execute(
            "SELECT ac.mix_mode, a.synthetic, a.ai_effect FROM timeline_audio_clips ac "
            "LEFT JOIN media_assets a ON a.id = ac.asset_id WHERE ac.timeline_id = ?",
            (timeline_id,),
        ).fetchall():
            if synth:
                arten.add("stimme" if effekt == "voiceover" else "synthetisch")
            elif mix in {"replace_original", "mute_original"}:
                arten.add("synthetisch")
    return arten


def _timeline_srt_segments(
    db: Database,
    timeline_id: str,
    tl: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return sequence-projected transcript segments suitable for SRT generation.

    Works for any timeline kind (sequence, rough_cut, scene) by walking the clips
    directly and projecting each segment's source-space range into sequence space.
    The integer affine transform is identical to the one used by
    :func:`sequence_transcript_blocks` and :func:`timeline_caption_words`.

    Returns an empty list when the timeline has no clips or no transcript data.
    """
    if tl["kind"] == "sequence":
        # For sequences, use the dedicated projection that walks the flattened view.
        return sequence_transcript_blocks(db, timeline_id)

    # For rough_cut / scene timelines, walk the lane-0 base clips directly.
    clips = [
        c for c in repos.list_timeline_clips(db, timeline_id)
        if int(c.get("lane") or 0) == 0 and str(c.get("role") or "base") == "base"
    ]
    clips.sort(key=lambda c: int(c["seq_in_frame"]))

    transcript_cache: dict[str, list[dict[str, Any]]] = {}
    segments: list[dict[str, Any]] = []

    for clip in clips:
        asset_id = str(clip["asset_id"])
        src_in = int(clip["src_in_frame"])
        src_out = int(clip["src_out_frame_exclusive"])
        seq_in = int(clip["seq_in_frame"])

        if asset_id not in transcript_cache:
            run = repos.get_latest_transcript_run(db, asset_id)
            transcript_cache[asset_id] = (
                repos.get_transcript(db, asset_id, run["id"]) if run is not None else []
            )

        for seg in transcript_cache[asset_id]:
            seg_start = int(seg["start_frame"])
            seg_end = int(seg["end_frame"])
            overlap_start = max(seg_start, src_in)
            overlap_end = min(seg_end, src_out)
            if overlap_start >= overlap_end:
                continue
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            segments.append(
                {
                    "seq_in_frame": seq_in + (overlap_start - src_in),
                    "seq_out_frame_exclusive": seq_in + (overlap_end - src_in),
                    "text": text,
                }
            )

    segments.sort(key=lambda s: (int(s["seq_in_frame"]), int(s["seq_out_frame_exclusive"])))
    return segments


def handle_render(ctx: JobContext) -> dict[str, Any]:
    """Claim an export row, resolve its timeline clips, run ffmpeg, mark done."""
    export_id = ctx.payload["export_id"]
    exp = repos.get_export(ctx.db, export_id)
    if exp is None:
        raise ValueError(f"export not found: {export_id}")

    tl = repos.get_timeline(ctx.db, exp["timeline_id"])
    if tl is None:
        repos.set_export_error(ctx.db, export_id, "timeline not found")
        raise ValueError("timeline not found")

    project = repos.get_project(ctx.db, exp["project_id"])
    if project is None:
        repos.set_export_error(ctx.db, export_id, "project not found")
        raise ValueError("project not found")

    clip_rows = resolve_clip_rows(ctx.db, tl)

    # Resolve asset paths for all clip rows (shared by both single- and multi-lane paths).
    resolved: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for c in clip_rows:
        a = repos.get_asset(ctx.db, c["asset_id"])
        if a is None:
            repos.set_export_error(ctx.db, export_id, f"asset not found: {c['asset_id']}")
            raise ValueError(f"asset not found: {c['asset_id']}")
        resolved.append((c, a))

    # Detect multi-lane: clips with lane > 0 that are NOT role="replace" trigger the overlay
    # compositor. role="replace" clips are already composited by apply_overlay_precedence (they
    # flatten to lane-0-equivalent for rendering purposes) and must NOT activate the multi-lane
    # path — that would break existing synthetic-disclosure and overlay tests.
    max_lane = max(
        (
            int(c.get("lane") or 0)
            for c, _ in resolved
            if str(c.get("role") or "base") != "replace"
        ),
        default=0,
    )
    is_multilane = max_lane > 0

    # Build the flat clip list for the single-lane path (original behavior: ALL resolved clips,
    # regardless of lane, passed to render_clips_mp4 in seq order).  Preserves backward
    # compatibility for role="replace" overlays (lane=1 from apply_overlay_precedence
    # but rendered as sequential clips by the existing concat path).
    # For caption/SRT/music, the code below already filters lane-0 base clips separately.
    clips: list[tuple[Path, int, int]] = [
        (Path(a["source_path"]), int(c["src_in_frame"]), int(c["src_out_frame_exclusive"]))
        for c, a in resolved
    ]

    # Reel duration cap: keep only the first N seconds (deterministic tail-trim). Captions, music
    # and transitions reference seq positions from the start, so trimming the tail leaves the kept
    # portion aligned. No-op when unset or the cut is already within budget.
    # NOTE: duration cap only applies on the single-lane path (lane-0 clips). Multi-lane timelines
    # with overlay lanes are not capped here — the full timeline is composited.
    max_s = (exp.get("options") or {}).get("max_duration_seconds")
    if isinstance(max_s, int) and max_s > 0 and not is_multilane:
        budget = round(max_s * project["sequence_rate_num"] / project["sequence_rate_den"])
        total = sum(fout - fin for _src, fin, fout in clips)
        # Smart end (default): snap the cut down to the latest transcript word boundary so the reel
        # finishes on a complete word, not mid-word. Still prefix-preserving (kept clips start at
        # seq 0), so alignment is untouched. Only consulted when a cut is actually needed; opt out
        # with reel_exact_duration=true to hit the exact frame budget.
        if total > budget and not (exp.get("options") or {}).get("reel_exact_duration"):
            word_ends = [w[2] for w in timeline_caption_words(ctx.db, exp["timeline_id"])]
            budget = snap_budget_to_word_boundary(budget, total, word_ends)
        clips = cap_clips_to_frames(clips, budget)

    if not clips:
        repos.set_export_error(ctx.db, export_id, "timeline has no clips")
        raise ValueError("no clips")

    # Build per-track music list:
    # - sequence timeline  → walk every scene; each with music contributes one track
    #   at its place in the assembled sequence.
    # - scene timeline     → at most one track covering the whole scene.
    # - other kinds        → no music.
    music_tracks: list[tuple[Path, int, int, int]] = []
    if tl["kind"] == "sequence":
        music_tracks = sequence_music_tracks(ctx.db, exp["timeline_id"])
    else:
        scene = repos.get_scene_by_timeline(ctx.db, exp["timeline_id"])
        if scene is not None and scene.get("music_asset_id"):
            masset = repos.get_asset(ctx.db, scene["music_asset_id"])
            if masset is not None:
                # Scene occupies [0, total_frames) — total = max seq_out across clip rows.
                total_frames = max(
                    (c["seq_out_frame_exclusive"] for c in clip_rows),
                    default=0,
                )
                music_tracks = [
                    (
                        Path(masset["source_path"]),
                        0,
                        total_frames,
                        int(scene["music_gain_percent"]),
                    )
                ]

    audio_overlays = _timeline_audio_overlays(ctx.db, exp["timeline_id"])
    video_transitions = (
        _sequence_video_transitions(ctx.db, exp["timeline_id"])
        if tl["kind"] == "sequence"
        else _clip_video_transitions(ctx.db, exp["timeline_id"])
    )

    opts: dict[str, object] = exp.get("options") or {}

    # Force disclosure overlay when the timeline has synthetic content and the caller
    # did not already supply a value. An explicit value (even blank) goes through
    # _effective_disclosure unchanged — only None triggers the auto-force.
    disc_text: str | None = opts.get("disclosure_text")  # type: ignore[assignment]
    if opts.get("kennzeichnung") == "auto":
        # Hinweis nach Inhalt (Video-Claw, 17.10.2026): Text aus den KI-Arten, None ohne KI.
        disc_text = kennzeichnung_text(_synthetic_kinds(ctx.db, exp["timeline_id"]))
    elif disc_text is None and _timeline_is_synthetic(ctx.db, exp["timeline_id"], audio_overlays):
        disc_text = _DEFAULT_DISCLOSURE

    caption_ass: str | None = None
    if opts.get("captions"):
        # caption_source (spec §5): "voiceover" -> narration-authored words only (never
        # falls back -- an empty result means no captions, matching "transcript"'s
        # equally strict behavior); "transcript" -> ASR words only; "auto" (default) ->
        # prefer voiceover words, fall back to transcript only when voiceover is empty
        # (e.g. no voiceover clips, or clips without a sidecar).
        source_mode = _option_str(
            opts, "caption_source", "auto", {"auto", "voiceover", "transcript"}
        )
        if source_mode == "voiceover":
            words = voiceover_caption_words(ctx.db, exp["timeline_id"])
        elif source_mode == "transcript":
            words = timeline_caption_words(ctx.db, exp["timeline_id"])
        else:
            words = voiceover_caption_words(ctx.db, exp["timeline_id"])
            if not words:
                words = timeline_caption_words(ctx.db, exp["timeline_id"])
        lines = group_caption_lines(words)
        if lines:
            preset = _option_str(
                opts, "caption_preset", "reels", {"reels", "tiktok", "shorts", "wide"}
            )
            play_w, play_h = _CAPTION_PRESETS[preset]
            # "wide" (16:9, landscape) reads better with a smaller relative fontsize and
            # margin than the portrait presets' 72/250 -- only applied when the caller
            # did not explicitly set caption_fontsize/caption_safe_margin.
            default_fontsize, default_margin_v = (54, 48) if preset == "wide" else (72, 250)
            caption_ass = build_ass(
                lines,
                rate_num=project["sequence_rate_num"],
                rate_den=project["sequence_rate_den"],
                play_w=play_w,
                play_h=play_h,
                fontsize=_option_int(opts, "caption_fontsize", default_fontsize, lo=24, hi=160),
                margin_v=_option_int(opts, "caption_safe_margin", default_margin_v, lo=0, hi=800),
                mode=_option_str(opts, "caption_mode", "karaoke", {"karaoke", "normal"}),
                position=_option_str(
                    opts, "caption_position", "bottom", {"top", "middle", "bottom"}
                ),
            )

    # Burn plain SRT captions into the standard MP4 export (non-reel path).
    # Only activated when burn_captions=True and no ASS captions are already present
    # (ASS captions are the reel-grade styled variant; SRT is the plain subtitle path).
    caption_srt: str | None = None
    if opts.get("burn_captions") and not caption_ass:
        srt_segments = _timeline_srt_segments(ctx.db, exp["timeline_id"], tl)
        if srt_segments:
            srt_text = sequence_transcript_to_srt(
                srt_segments,
                rate_num=project["sequence_rate_num"],
                rate_den=project["sequence_rate_den"],
            )
            caption_srt = srt_text or None

    dest = Path(project["workspace_root"]) / "exports" / f"{export_id}.mp4"
    try:
        if is_multilane:
            # Multi-lane compositor path: build per-lane clip tuples and call the overlay renderer.
            # Captions/reel features are single-lane only (v1 scope); they are not threaded here.
            # Exclude role="replace" clips — they are flattened by apply_overlay_precedence and
            # are not free-placed multi-lane overlay clips.
            lane_clip_rows: list[list[tuple[Path, int, int, int, int]]] = [
                [] for _ in range(max_lane + 1)
            ]
            for c, a in resolved:
                if str(c.get("role") or "base") == "replace":
                    continue
                lane = int(c.get("lane") or 0)
                lane_clip_rows[lane].append((
                    Path(a["source_path"]),
                    int(c["src_in_frame"]),
                    int(c["src_out_frame_exclusive"]),
                    int(c["seq_in_frame"]),
                    int(c["seq_out_frame_exclusive"]),
                ))
            # Sort each lane by seq_in.
            for lane_list in lane_clip_rows:
                lane_list.sort(key=lambda t: t[3])
            render_multilane_mp4(
                lane_clip_rows,
                dest,
                rate_num=project["sequence_rate_num"],
                rate_den=project["sequence_rate_den"],
                music_tracks=music_tracks if music_tracks else None,
                audio_overlays=audio_overlays if audio_overlays else None,
                video_transitions=video_transitions if video_transitions else None,
            )
            # expected_frames for multi-lane: max seq_out across all clip rows.
            expected_frames = max(
                (int(c["seq_out_frame_exclusive"]) for c, _ in resolved),
                default=0,
            )
        else:
            render_clips_mp4(
                clips,
                dest,
                rate_num=project["sequence_rate_num"],
                rate_den=project["sequence_rate_den"],
                music_tracks=music_tracks if music_tracks else None,
                audio_overlays=audio_overlays if audio_overlays else None,
                vertical=bool(opts.get("vertical", False)),
                hook_text=opts.get("hook_text"),  # type: ignore[arg-type]
                disclosure_text=disc_text,
                caption_ass=caption_ass,
                caption_srt=caption_srt,
                video_transitions=video_transitions if video_transitions else None,
            )
            expected_frames = sum(fout - fin for _src, fin, fout in clips)
        assert_or_fix_media_sync(
            dest,
            expected_frames=expected_frames,
            rate_num=project["sequence_rate_num"],
            rate_den=project["sequence_rate_den"],
            require_video=True,
            fix=True,
        )
        size_bytes = os.path.getsize(dest)
        max_bytes = opts.get("max_bytes")
        if isinstance(max_bytes, int) and max_bytes > 0 and size_bytes > max_bytes:
            # Groessengrenze (Video-Claw): neu kodieren oder mit Grund scheitern (except unten).
            dauer_s = expected_frames * project["sequence_rate_den"] / project["sequence_rate_num"]
            size_bytes = groesse.einhalten(dest, max_bytes=max_bytes, dauer_s=dauer_s)
    except Exception as e:  # noqa: BLE001 - persist the failure, drop partial output, re-raise
        repos.set_export_error(ctx.db, export_id, str(e)[-500:])
        if dest.exists():
            dest.unlink(missing_ok=True)
        record_short_run_result(
            ctx.db,
            exp["options"],
            status="failed",
            trace={"error": str(e)[-500:]},
        )
        raise

    repos.set_export_done(ctx.db, export_id, path=str(dest), size_bytes=size_bytes)
    # Best-effort copy into object storage so other machines can fetch the video
    # instead of being handed a path that exists only here. `publish` never raises:
    # the export above is already 'ready', and an unreachable bucket must not undo
    # an hour of rendering.
    schluessel = publish(store_from_env(), f"exports/{export_id}{dest.suffix}", dest)
    if schluessel is not None:
        repos.set_export_object(ctx.db, export_id, schluessel)
    try:
        audit.record(
            ctx.db,
            audit.system_principal(),
            "export.render",
            entity_type="export",
            entity_id=export_id,
            payload={
                "timeline_id": exp.get("timeline_id"),
                "format": "mp4",
                "path": str(dest),
                "size_bytes": size_bytes,
            },
        )
    except Exception:  # noqa: BLE001
        _log.exception("audit.record failed for export_id=%s (ignored)", export_id)
    record_short_run_result(
        ctx.db,
        exp["options"],
        status="succeeded",
        trace={"export_id": export_id, "output": str(dest)},
    )
    return {"export_id": export_id, "path": str(dest)}


def register_render_handlers(registry: dict[str, JobHandler]) -> None:
    """Register all render-stage job handlers into ``registry``."""
    registry["export.render"] = handle_render
