"""Job handler: ai.reenact — consent-gated portrait reenactment.

SAFETY-CRITICAL: the consent gate is checked FIRST, before any DB writes or
file I/O.  If the consent record is missing or the payload is malformed the
handler raises immediately and creates nothing.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from .. import audit
from ..db import repos
from ..db.database import Database
from ..editing.history import perform_undo, timeline_checkpoint
from ..editing.operations import EditClip, append_clip, ordered
from ..ingest.ffmpeg import probe as probe_media
from ..jobs.keys import idempotency_key_for
from ..jobs.queues import queue_for
from ..jobs.runner import JobContext, JobHandler, enqueue
from ..render.mp4 import render_clips_mp4
from ..render.sync import assert_or_fix_media_sync
from ..sequences.flatten import flatten_sequence
from ..util import new_id
from .auto_pipeline import plan_lipsync_after_voiceover
from .lipsync_backend import resolve_lipsync_backend
from .provenance import write_ai_provenance_manifest
from .reenact_backend import resolve_reenact_backend
from .vo_words import write_word_sidecar
from .voiceover_backend import DEFAULT_VOICEOVER_SAMPLE_RATE, resolve_voiceover_backend

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeBackendConfig:
    name: str | None
    base_url: str | None
    runtime_id: str | None = None


_EFFECT_BACKENDS = {
    "voice": "sidecar",
    "reenact": "liveportrait",
    "lipsync": "vibevideo",
}

# Natural-mode voiceover hint cap (Task 5 narrated-reel, I-3): `duration_frames` is a hint,
# never a target (see `_synthesize_voiceover_asset`'s docstring), but `StubVoiceoverBackend`
# takes it literally and writes exactly that many frames of audio regardless. Without a cap,
# hinting with the full remaining source-asset length (often minutes) makes the stub backend
# synthesize a multi-megabyte WAV per beat that then clips straight back down to the asset
# end anyway. Real backends (SAPI/EL/Chatterbox) ignore the hint in natural mode, so this cap
# only changes stub-backend behavior.
_NATURAL_MODE_HINT_CAP_SECONDS = 90


def _runtime_base_url(runtime: dict[str, Any]) -> str | None:
    raw = runtime.get("base_url")
    if isinstance(raw, str) and raw.strip():
        return raw.rstrip("/")
    port = runtime.get("port")
    if port is not None:
        return f"http://127.0.0.1:{int(port)}"
    return None


def _backend_config_from_runtime(
    db: Database,
    runtime_id: str | None,
    fallback: str | None,
    effect: str,
) -> RuntimeBackendConfig:
    if runtime_id is None:
        return RuntimeBackendConfig(fallback, None, None)

    runtime = repos.get_ai_runtime(db, runtime_id)
    if runtime is None:
        raise ValueError(f"runtime not found: {runtime_id!r}")
    if not bool(runtime["enabled"]):
        raise ValueError("runtime is disabled")
    if str(runtime["effect"]) != effect:
        raise ValueError(f"runtime effect must be {effect}")

    kind = str(runtime["kind"])
    if kind == "stub":
        return RuntimeBackendConfig("stub", None, runtime_id)
    if kind in {"external_http", "container"}:
        base_url = _runtime_base_url(runtime)
        if base_url is None:
            raise ValueError("runtime has no base_url or port")
        return RuntimeBackendConfig(_EFFECT_BACKENDS[effect], base_url, runtime_id)
    raise ValueError(f"unsupported runtime kind: {kind!r}")


def _backend_from_runtime(
    db: Database,
    runtime_id: str | None,
    fallback: str | None,
    effect: str | None = None,
) -> str | None:
    if effect is None:
        if runtime_id is None:
            return fallback
        runtime = repos.get_ai_runtime(db, runtime_id)
        if runtime is None:
            raise ValueError(f"runtime not found: {runtime_id!r}")
        effect = str(runtime["effect"])
    return _backend_config_from_runtime(db, runtime_id, fallback, effect).name


def handle_reenact(ctx: JobContext) -> dict[str, Any]:
    """Consent-gated reenact job.

    Steps (in order):
    1. Consent gate — raises immediately if consent_id is missing or unknown.
    2. Resolve timeline + project (rate_num / rate_den).
    3. Extract the driving range from base clip rows that overlap [seq_in, seq_out).
    4. Render the driving clip to a temporary MP4.
    5. Resolve and check the reenact backend.
    6. Animate the portrait asset, write to the project workspace.
    7. Register the output as a synthetic asset.
    8. Place a replace-overlay clip on the timeline.
    """
    payload = ctx.payload

    # ── 1. CONSENT GATE (must be first — creates nothing on failure) ──────────
    consent_id: str | None = payload.get("consent_id")
    if not consent_id:
        raise ValueError("ai.reenact: payload missing required key 'consent_id'")

    consent = repos.get_consent_record(ctx.db, consent_id)
    if consent is None:
        raise ValueError(
            f"ai.reenact: consent record not found: {consent_id!r} — "
            "refusing to create any asset or clip"
        )
    if consent.get("revoked_at"):
        raise ValueError(
            f"ai.reenact: consent {consent_id!r} has been revoked — "
            "refusing to create any asset or clip"
        )

    # ── 2. Resolve timeline + project ────────────────────────────────────────
    timeline_id: str = payload["timeline_id"]
    tl = repos.get_timeline(ctx.db, timeline_id)
    if tl is None:
        raise ValueError(f"ai.reenact: timeline not found: {timeline_id!r}")

    project = repos.get_project(ctx.db, tl["project_id"])
    if project is None:
        raise ValueError(f"ai.reenact: project not found: {tl['project_id']!r}")

    rate_num: int = int(project["sequence_rate_num"])
    rate_den: int = int(project["sequence_rate_den"])

    seq_in: int = int(payload["seq_in_frame"])
    seq_out: int = int(payload["seq_out_frame_exclusive"])

    # ── 3. Build the driving clip list from overlapping ORIGINAL base rows ────
    # Drive from original base footage ONLY — never from a prior synthetic
    # replace-overlay (provenance: a reenact must not be driven by another
    # reenact's output). So we resolve base rows directly, WITHOUT precedence.
    if tl.get("kind") == "sequence":
        base_rows = flatten_sequence(ctx.db, tl["id"])
    else:
        base_rows = [
            r
            for r in repos.list_timeline_clips(ctx.db, tl["id"])
            if r.get("role", "base") != "replace"
        ]

    driving_clips: list[tuple[Path, int, int]] = []
    for row in base_rows:
        row_seq_in = int(row["seq_in_frame"])
        row_seq_out = int(row["seq_out_frame_exclusive"])
        row_src_in = int(row["src_in_frame"])

        o_in = max(row_seq_in, seq_in)
        o_out = min(row_seq_out, seq_out)
        if o_in >= o_out:
            continue  # no overlap

        s_in = row_src_in + (o_in - row_seq_in)
        s_out = row_src_in + (o_out - row_seq_in)

        asset = repos.get_asset(ctx.db, row["asset_id"])
        if asset is None:
            raise ValueError(f"ai.reenact: asset not found: {row['asset_id']!r}")

        driving_clips.append((Path(asset["source_path"]), s_in, s_out))

    if not driving_clips:
        raise ValueError(
            f"ai.reenact: no base clips overlap driving range [{seq_in}, {seq_out})"
        )

    # ── 4. Render the driving range to a temp file ────────────────────────────
    workspace = Path(project["workspace_root"])
    tmp_dir = workspace / "tmp"
    synthetic_dir = workspace / "synthetic"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    # Bind both paths up front so out_path can be cleaned on ANY failure below
    # (no orphaned, unlabelled partial synthetic file may ever survive a crash).
    driving_tmp = tmp_dir / f"{new_id()}.driving.mp4"
    out_path = synthetic_dir / f"{new_id()}.mp4"
    try:
        render_clips_mp4(
            driving_clips,
            driving_tmp,
            rate_num=rate_num,
            rate_den=rate_den,
        )

        # ── 5. Resolve and validate backend ──────────────────────────────────
        backend_config = _backend_config_from_runtime(
            ctx.db,
            payload.get("runtime_id"),
            payload.get("backend"),
            "reenact",
        )
        backend = resolve_reenact_backend(
            backend_config.name,
            base_url=backend_config.base_url,
        )
        if not backend.available():
            raise RuntimeError(
                f"ai.reenact: reenact backend '{backend.name}' is not installed"
            )

        # ── 6. Portrait asset → animate ──────────────────────────────────────
        portrait_asset_id: str = payload["portrait_asset_id"]
        portrait = repos.get_asset(ctx.db, portrait_asset_id)
        if portrait is None:
            raise ValueError(
                f"ai.reenact: portrait asset not found: {portrait_asset_id!r}"
            )

        backend.reenact(
            driving_path=driving_tmp,
            portrait_path=Path(portrait["source_path"]),
            out_path=out_path,
            fps_num=rate_num,
            fps_den=rate_den,
        )
        assert_or_fix_media_sync(
            out_path,
            expected_frames=seq_out - seq_in,
            rate_num=rate_num,
            rate_den=rate_den,
            require_video=True,
            fix=True,
        )
    except Exception:
        # Never leave an unlabelled partial synthetic file behind on failure.
        out_path.unlink(missing_ok=True)
        raise
    finally:
        # The temporary driving file is never needed after this block.
        driving_tmp.unlink(missing_ok=True)

    # ── 7. Register the output as a synthetic asset ───────────────────────────
    asset = repos.create_asset(
        ctx.db,
        project_id=tl["project_id"],
        type="video",
        display_name=f"reenact {seq_in}-{seq_out}",
        source_path=str(out_path),
        synthetic=True,
        ai_effect="reenact",
    )
    write_ai_provenance_manifest(
        media_path=out_path,
        asset_id=asset["id"],
        project_id=tl["project_id"],
        ai_effect="reenact",
        source={
            "timeline_id": tl["id"],
            "portrait_asset_id": portrait_asset_id,
            "consent_id": consent_id,
            "backend": payload.get("backend") or "stub",
            "seq_in_frame": seq_in,
            "seq_out_frame_exclusive": seq_out,
        },
    )

    # ── 8. Place a replace-overlay clip on the timeline ───────────────────────
    if repos.is_job_cancel_requested(ctx.db, ctx.job_id):
        return {"status": "cancelled", "reason": "undo"}
    repos.add_timeline_clip(
        ctx.db,
        timeline_id=tl["id"],
        asset_id=asset["id"],
        src_in_frame=0,
        src_out_frame_exclusive=seq_out - seq_in,
        seq_in_frame=seq_in,
        seq_out_frame_exclusive=seq_out,
        lane=1,
        role="replace",
    )

    try:
        audit.record(
            ctx.db,
            audit.system_principal(),
            "ai.reenact",
            entity_type="media_asset",
            entity_id=asset["id"],
            payload={
                "timeline_id": tl["id"],
                "consent_id": consent_id,
                "seq_in_frame": seq_in,
                "seq_out_frame_exclusive": seq_out,
            },
        )
    except Exception:
        _log.warning("audit.record failed for ai.reenact; job result preserved", exc_info=True)

    return {
        "asset_id": asset["id"],
        "out_path": str(out_path),
        "seq_in_frame": seq_in,
        "seq_out_frame_exclusive": seq_out,
    }


def register_ai_handlers(registry: dict[str, JobHandler]) -> None:
    """Register all AI-stage job handlers into ``registry``."""
    registry["ai.reenact"] = handle_reenact
    registry["ai.voiceover"] = handle_voiceover
    registry["ai.lipsync"] = handle_lipsync
    registry["ai.narrated_reel"] = handle_narrated_reel


def _maybe_enqueue_lipsync_after_vo(
    ctx: JobContext,
    *,
    timeline: dict[str, Any],
    project: dict[str, Any],
    vo_asset_id: str,
    seq_in: int,
    seq_out: int,
) -> tuple[str | None, str | None]:
    """Probe the VO span for a face and, if consent exists, enqueue ai.lipsync (spec §5).

    No face / sidecar absent / no consent -> (None, reason); never raises into the VO
    success path (a lipsync hiccup must not fail the VO that already rendered).
    """
    try:
        backend = resolve_lipsync_backend(None)
        if not backend.available():
            return None, "no_backend"
        rate_num = int(project["sequence_rate_num"])
        rate_den = int(project["sequence_rate_den"])
        if timeline.get("kind") == "sequence":
            base_rows = flatten_sequence(ctx.db, timeline["id"])
        else:
            base_rows = [
                r for r in repos.list_timeline_clips(ctx.db, timeline["id"])
                if r.get("role", "base") != "replace" and int(r.get("lane") or 0) == 0
            ]
        # Restrict probe base rows to the primary asset in the VO span so we don't
        # stitch footage from different sources into the probe clip (multi-asset cut).
        primary_asset_id: str | None = None
        for _br in base_rows:
            r_in = int(_br["seq_in_frame"])
            r_out = int(_br["seq_out_frame_exclusive"])
            if min(r_out, seq_out) > max(r_in, seq_in):  # overlaps VO span
                primary_asset_id = str(_br["asset_id"])
                break
        if primary_asset_id is not None:
            base_rows = [r for r in base_rows if str(r["asset_id"]) == primary_asset_id]
        clips: list[tuple[Path, int, int]] = []
        for row in base_rows:
            r_in, r_out = int(row["seq_in_frame"]), int(row["seq_out_frame_exclusive"])
            o_in, o_out = max(r_in, seq_in), min(r_out, seq_out)
            if o_in >= o_out:
                continue
            asset = repos.get_asset(ctx.db, row["asset_id"])
            if asset is None:
                continue
            base = int(row["src_in_frame"])
            clips.append((Path(asset["source_path"]), base + (o_in - r_in), base + (o_out - r_in)))
        if not clips:
            return None, "no_face"
        tmp_dir = Path(project["workspace_root"]) / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        driving = tmp_dir / f"{new_id()}.vo-probe.mp4"
        vo_asset = repos.get_asset(ctx.db, vo_asset_id)
        audio_path = Path(vo_asset["source_path"]) if vo_asset else None
        try:
            render_clips_mp4(clips, driving, rate_num=rate_num, rate_den=rate_den)
            probe = backend.probe(video_path=driving, audio_path=audio_path or driving)
        finally:
            driving.unlink(missing_ok=True)
        consent_id = repos.get_active_consent_id(ctx.db, project["id"])
        plan = plan_lipsync_after_voiceover(
            probe_face_detected=bool(probe.face_detected),
            probe_mouth_visible=bool(probe.mouth_visible),
            consent_id=consent_id,
            audio_asset_id=vo_asset_id,
            seq_in_frame=seq_in,
            seq_out_frame_exclusive=seq_out,
        )
        if not plan.should_enqueue:
            return None, plan.reason
        payload: dict[str, Any] = {
            "timeline_id": timeline["id"],
            "seq_in_frame": plan.seq_in_frame,
            "seq_out_frame_exclusive": plan.seq_out_frame_exclusive,
            "audio_asset_id": plan.audio_asset_id,
            "consent_id": plan.consent_id,
            "license_accepted": True,
        }
        job_id = enqueue(
            ctx.db,
            queue=queue_for("ai.lipsync", default="ai"),
            kind="ai.lipsync",
            payload=payload,
            max_attempts=2,
            idempotency_key=idempotency_key_for("ai.lipsync", payload),
            caused_by_job_id=ctx.job_id,
        )
        return job_id, "ok"
    except Exception:  # noqa: BLE001 - VO already succeeded; lipsync is best-effort
        _log.warning(
            "lipsync auto-enqueue failed after voiceover; VO result preserved",
            exc_info=True,
        )
        return None, "probe_error"


def _measure_wav_frames_ceil(path: Path, *, rate_num: int, rate_den: int) -> int:
    """The WAV's real length in project frames per ffprobe, rounded UP.

    Natural-fit clips must never be a frame short of the speech they actually contain
    (spec §3) -- rounding up trades a few extra trailing samples for that guarantee
    instead of risking truncation from float/duration-string rounding.

    A missing, zero/negative, ``"N/A"`` (ffprobe's own sentinel for "unknown"), or
    otherwise unparseable duration means the synthesized WAV cannot be measured --
    treated as a hard failure rather than silently producing a zero/near-zero-length
    clip that would still report job success. Mirrors the ``"N/A"`` handling in
    ``render.sync._duration_seconds``.
    """
    data = probe_media(path)
    raw_duration = data.get("format", {}).get("duration")
    if raw_duration is None:
        streams = data.get("streams") or []
        audio_stream = next(
            (s for s in streams if isinstance(s, dict) and s.get("codec_type") == "audio"),
            None,
        )
        raw_duration = audio_stream.get("duration") if audio_stream else None
    duration: Fraction | None = None
    if isinstance(raw_duration, str) and raw_duration and raw_duration != "N/A":
        try:
            duration = Fraction(raw_duration)
        except ValueError:
            duration = None
    elif isinstance(raw_duration, (int, float)):
        duration = Fraction(str(raw_duration))
    if duration is None or duration <= 0:
        raise RuntimeError(
            f"ai.voiceover: synthesized WAV has no measurable duration ({path})"
        )
    frames = duration * Fraction(rate_num, rate_den)
    return math.ceil(frames)


def _synthesize_voiceover_asset(
    ctx: JobContext,
    *,
    project: dict[str, Any],
    timeline: dict[str, Any],
    text: str,
    backend_config: RuntimeBackendConfig,
    language: str | None,
    voice_id: str | None,
    duration_frames: int,
    fit_to_slot: bool,
    sample_rate: int = DEFAULT_VOICEOVER_SAMPLE_RATE,
    provenance_source: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int, Path]:
    """Synthesize one voiceover WAV, register it as a project asset, and probe it.

    Shared building block for ``handle_voiceover`` (both fit modes) and the future
    narrated-reel collage job (spec §6), so both paths produce an identical asset /
    provenance / probe shape.

    ``duration_frames`` is always the caller's SLOT hint (never the desired natural
    length) -- backends that need a budget get one either way; only ``fit_to_slot``
    changes what they do with it.

    * ``fit_to_slot=True`` (slot mode): the backend pads/trims to exactly
      ``duration_frames`` and the existing sync guard normalizes any drift, unchanged
      from before this task. The returned ``measured_frames`` equals ``duration_frames``.
    * ``fit_to_slot=False`` (natural mode): the backend writes the full natural-length
      speech; the real WAV duration is then measured via ffprobe and rounded UP
      (:func:`_measure_wav_frames_ceil`). No sync-fix runs -- the measured length IS the
      clip's true length, there is nothing to normalize against.

    Returns ``(asset_row, measured_frames, out_path)``.
    """
    rate_num = int(project["sequence_rate_num"])
    rate_den = int(project["sequence_rate_den"])

    backend = resolve_voiceover_backend(backend_config.name, base_url=backend_config.base_url)
    if not backend.available():
        raise RuntimeError(f"ai.voiceover: voiceover backend '{backend.name}' is not installed")

    workspace = Path(project["workspace_root"])
    synthetic_dir = workspace / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    out_path = synthetic_dir / f"{new_id()}.voiceover.wav"
    try:
        backend.synthesize(
            text=text,
            out_path=out_path,
            duration_frames=duration_frames,
            fps_num=rate_num,
            fps_den=rate_den,
            sample_rate=sample_rate,
            language=language,
            voice_id=voice_id,
            fit_to_slot=fit_to_slot,
        )
        if fit_to_slot:
            measured_frames = duration_frames
            assert_or_fix_media_sync(
                out_path,
                expected_frames=duration_frames,
                rate_num=rate_num,
                rate_den=rate_den,
                require_audio=True,
                fix=True,
            )
        else:
            measured_frames = _measure_wav_frames_ceil(
                out_path, rate_num=rate_num, rate_den=rate_den
            )
    except Exception:
        out_path.unlink(missing_ok=True)
        raise

    asset = repos.create_asset(
        ctx.db,
        project_id=timeline["project_id"],
        type="audio",
        display_name="Voiceover",
        source_path=str(out_path),
        synthetic=True,
        ai_effect="voiceover",
    )
    repos.add_asset_file(
        ctx.db,
        asset_id=asset["id"],
        kind="original",
        path=str(out_path),
        size_bytes=out_path.stat().st_size,
        is_proxy=False,
    )
    source: dict[str, Any] = {
        "timeline_id": timeline["id"],
        "backend": backend.name,
        "voice_id": voice_id,
        "language": language,
    }
    if provenance_source:
        source.update(provenance_source)
    write_ai_provenance_manifest(
        media_path=out_path,
        asset_id=asset["id"],
        project_id=timeline["project_id"],
        ai_effect="voiceover",
        source=source,
    )
    repos.update_asset_probe(
        ctx.db,
        asset["id"],
        type="audio",
        duration_frames=measured_frames,
        rate_num=rate_num,
        rate_den=rate_den,
        audio_sample_rate=sample_rate,
        start_timecode=None,
        width=None,
        height=None,
        codec_video=None,
        codec_audio="pcm_s16le",
        is_vfr=False,
        sha256=None,
    )

    # Word-timing sidecar for narration captions (spec §4). Applies to both fit modes
    # (slot: measured_frames == duration_frames; natural: the real measured length) and
    # is never fatal -- a sidecar problem must not affect this job's result.
    write_word_sidecar(
        out_path,
        text=text,
        measured_frames=measured_frames,
        rate_num=rate_num,
        rate_den=rate_den,
        language=language,
    )

    return asset, measured_frames, out_path


def handle_voiceover(ctx: JobContext) -> dict[str, Any]:
    """Generate a synthetic voiceover WAV and place it on the timeline's A2 lane.

    ``fit="slot"`` (default): pads/trims to the requested span -- unchanged behavior.
    ``fit="natural"``: the clip span is derived from the measured speech length instead;
    the requested ``seq_out_frame_exclusive`` acts only as an UPPER BOUND (spec §3), so
    the clip can end earlier than requested but never later.
    """
    payload = ctx.payload
    timeline_id = str(payload["timeline_id"])
    timeline = repos.get_timeline(ctx.db, timeline_id)
    if timeline is None:
        raise ValueError(f"ai.voiceover: timeline not found: {timeline_id!r}")

    project = repos.get_project(ctx.db, timeline["project_id"])
    if project is None:
        raise ValueError(f"ai.voiceover: project not found: {timeline['project_id']!r}")

    seq_in = int(payload["seq_in_frame"])
    seq_out_requested = int(payload["seq_out_frame_exclusive"])
    if seq_out_requested <= seq_in:
        raise ValueError("ai.voiceover: seq_out_frame_exclusive must be greater than seq_in_frame")

    text = str(payload.get("text") or "").strip()
    segment_id = payload.get("segment_id")
    if segment_id:
        segment = repos.get_segment(ctx.db, str(segment_id))
        if segment is None:
            raise ValueError(f"ai.voiceover: segment not found: {segment_id!r}")
        source_asset = repos.get_asset(ctx.db, segment["asset_id"])
        if source_asset is None:
            raise ValueError(f"ai.voiceover: segment asset not found: {segment['asset_id']!r}")
        if source_asset["project_id"] != timeline["project_id"]:
            raise ValueError("ai.voiceover: segment does not belong to this timeline project")
        if not text:
            text = str(segment["text"]).strip()
    if not text:
        raise ValueError("ai.voiceover: text is required")

    slot_frames = seq_out_requested - seq_in
    fit = str(payload.get("fit") or "slot")
    fit_to_slot = fit != "natural"
    pad_frames = int(payload["pad_frames"]) if payload.get("pad_frames") is not None else 12

    backend_config = _backend_config_from_runtime(
        ctx.db,
        payload.get("runtime_id"),
        payload.get("backend"),
        "voice",
    )

    asset, measured_frames, out_path = _synthesize_voiceover_asset(
        ctx,
        project=project,
        timeline=timeline,
        text=text,
        backend_config=backend_config,
        language=payload.get("language"),
        voice_id=payload.get("voice_id"),
        duration_frames=slot_frames,
        fit_to_slot=fit_to_slot,
        provenance_source={
            "segment_id": segment_id,
            "seq_in_frame": seq_in,
            "seq_out_frame_exclusive": seq_out_requested,
        },
    )

    if fit_to_slot:
        seq_out_eff = seq_out_requested
    else:
        seq_out_eff = min(seq_in + measured_frames + pad_frames, seq_out_requested)

    # Remove any prior synthetic VO clips overlapping this span so editing the same span
    # twice (different text or voice) never stacks two replace_original clips.
    if repos.is_job_cancel_requested(ctx.db, ctx.job_id):
        return {"status": "cancelled", "reason": "undo"}
    effective_mix_mode = str(payload.get("mix_mode") or "mix")
    if effective_mix_mode in {"replace_original", "mute_original"}:
        repos.delete_timeline_audio_clips_overlapping(
            ctx.db,
            timeline_id=timeline["id"],
            seq_in=seq_in,
            seq_out_excl=seq_out_eff,
            mix_mode=effective_mix_mode,
        )

    clip = repos.add_timeline_audio_clip(
        ctx.db,
        timeline_id=timeline["id"],
        asset_id=asset["id"],
        seq_in_frame=seq_in,
        seq_out_frame_exclusive=seq_out_eff,
        asset_in_frame=0,
        gain_percent=int(payload.get("gain_percent") or 100),
        fade_in_frames=int(payload.get("fade_in_frames") or 0),
        fade_out_frames=int(payload.get("fade_out_frames") or 0),
        mix_mode=effective_mix_mode,
        ducking_percent=(
            int(payload["ducking_percent"]) if payload.get("ducking_percent") is not None else 100
        ),
        label="Voiceover",
    )

    lipsync_job_id, lipsync_skip_reason = _maybe_enqueue_lipsync_after_vo(
        ctx,
        timeline=timeline,
        project=project,
        vo_asset_id=asset["id"],
        seq_in=seq_in,
        seq_out=seq_out_eff,
    )

    try:
        audit.record(
            ctx.db,
            audit.system_principal(),
            "ai.voiceover",
            entity_type="media_asset",
            entity_id=asset["id"],
            payload={
                "timeline_id": timeline["id"],
                "audio_clip_id": clip["id"],
                "seq_in_frame": seq_in,
                "seq_out_frame_exclusive": seq_out_eff,
            },
        )
    except Exception:
        _log.warning("audit.record failed for ai.voiceover; job result preserved", exc_info=True)

    return {
        "asset_id": asset["id"],
        "audio_clip_id": clip["id"],
        "out_path": str(out_path),
        "seq_in_frame": seq_in,
        "seq_out_frame_exclusive": seq_out_eff,
        "measured_frames": measured_frames,
        "lipsync_job_id": lipsync_job_id,
        "lipsync_skip_reason": lipsync_skip_reason,
    }


def handle_narrated_reel(ctx: JobContext) -> dict[str, Any]:
    """Collage-builder job (spec §6): a beat list -> a finished narrated-reel timeline.

    Runs the whole build inside ONE undo checkpoint (``timeline_checkpoint``): pushing it
    before the loop captures the empty pre-job timeline, so a single undo reverts the
    entire collage rather than one beat at a time.

    Per beat, sequentially:
      1. cancel-check FIRST (before any mutation for this beat) — cooperative cancellation
         must never leave a half-synthesized beat behind.
      2. synthesize a natural-length voiceover (:func:`_synthesize_voiceover_asset`,
         ``fit_to_slot=False``), hinted with the asset's remaining frames from ``src_in``,
         capped at ``_NATURAL_MODE_HINT_CAP_SECONDS`` (the hint is only a budget for backends
         that need one; a stub/placeholder backend takes it literally, so an uncapped hint on
         a long source asset would make it synthesize minutes of audio for nothing).
      3. the beat's clip length is the measured speech + ``pad_frames``, clamped so the
         clip never runs past the source asset's end (a clamp appends a warning).
      4. the video clip is appended (repos-level, same primitive as the operations
         endpoint's ``append_clip``) and the voice clip is placed ``replace_original``
         over the beat's span.

    Transitions (crossfade between every clip but the last, fade on the last) and the
    optional chained render are applied once, after all beats, against the FINAL stable
    clip ids — ``replace_timeline_clips`` reassigns every clip a fresh id on every call,
    so setting transitions mid-loop would silently be wiped out by the next beat's append.

    A mid-build exception (a beat's synthesis/backend failing, a DB error, ...) rolls the
    ENTIRE timeline back to its pre-job state via :func:`laura.editing.history.perform_undo`
    — the same machinery the ``/undo`` endpoint uses — before re-raising, so the job still
    fails with a clear error but never leaves a half-built collage (a partial run of clips
    with no matching voice track, or vice versa) sitting on the timeline (spec's
    Fehlerfälle: "Timeline bleibt im letzten konsistenten Zustand"). Cooperative
    cancellation is NOT an error path — it returns normally (see below), so already
    -completed beats stay committed on cancel, unlike on a genuine failure.
    """
    payload = ctx.payload
    timeline_id = str(payload["timeline_id"])
    timeline = repos.get_timeline(ctx.db, timeline_id)
    if timeline is None:
        raise ValueError(f"ai.narrated_reel: timeline not found: {timeline_id!r}")

    project = repos.get_project(ctx.db, timeline["project_id"])
    if project is None:
        raise ValueError(f"ai.narrated_reel: project not found: {timeline['project_id']!r}")

    beats_in = list(payload.get("beats") or [])
    crossfade_frames = int(payload.get("crossfade_frames") or 0)
    final_fade_frames = int(payload.get("final_fade_frames") or 0)
    language = payload.get("language")
    voice_id = payload.get("voice_id")
    backend_config = _backend_config_from_runtime(
        ctx.db, payload.get("runtime_id"), payload.get("backend"), "voice",
    )
    natural_hint_cap_frames = (
        _NATURAL_MODE_HINT_CAP_SECONDS
        * int(project["sequence_rate_num"])
        // int(project["sequence_rate_den"])
    )

    beats_result: list[dict[str, Any]] = []
    warnings: list[str] = []
    cancelled = False

    with timeline_checkpoint(ctx.db, timeline_id, "Narrated Reel erstellt"):
        try:
            edit_clips: list[EditClip] = [
                EditClip.from_row(c) for c in repos.list_timeline_clips(ctx.db, timeline_id)
            ]
            for i, beat in enumerate(beats_in):
                if repos.is_job_cancel_requested(ctx.db, ctx.job_id):
                    cancelled = True
                    break

                text = str(beat["text"])
                asset_id = str(beat["asset_id"])
                src_in = int(beat["src_in_frame"])
                pad_frames = (
                    int(beat["pad_frames"]) if beat.get("pad_frames") is not None else 12
                )

                asset = repos.get_asset(ctx.db, asset_id)
                if asset is None:
                    raise ValueError(
                        f"ai.narrated_reel: beat {i}: asset not found: {asset_id!r}"
                    )
                asset_duration = asset.get("duration_frames")
                if asset_duration is None:
                    raise ValueError(f"ai.narrated_reel: beat {i}: asset has no known duration")
                remaining = int(asset_duration) - src_in
                if remaining <= 0:
                    raise ValueError(
                        f"ai.narrated_reel: beat {i}: src_in_frame is at/after the asset end"
                    )

                # Natural-mode hint only -- never the desired length (see
                # `_synthesize_voiceover_asset`'s docstring). `remaining` (uncapped) still
                # governs the actual clip/clamp math below; only the SLOT HINT handed to the
                # backend is capped (I-3).
                hint_frames = min(remaining, natural_hint_cap_frames)
                voice_asset, measured_frames, _out_path = _synthesize_voiceover_asset(
                    ctx,
                    project=project,
                    timeline=timeline,
                    text=text,
                    backend_config=backend_config,
                    language=language,
                    voice_id=voice_id,
                    duration_frames=hint_frames,
                    fit_to_slot=False,
                    provenance_source={
                        "narrated_reel_timeline_id": timeline_id,
                        "narrated_reel_beat_index": i,
                        "asset_id": asset_id,
                    },
                )

                wanted_len = measured_frames + pad_frames
                beat_len = min(wanted_len, remaining)
                if wanted_len > remaining:
                    warnings.append(f"beat {i}: clipped to asset end")

                clip = EditClip(
                    asset_id=asset_id,
                    src_in_frame=src_in,
                    src_out_frame_exclusive=src_in + beat_len,
                    seq_in_frame=0,
                    seq_out_frame_exclusive=0,
                    lane=0,
                )
                edit_clips = ordered(append_clip(edit_clips, clip))
                repos.replace_timeline_clips(
                    ctx.db, timeline_id, [c.to_row() for c in edit_clips]
                )
                placed = edit_clips[-1]
                seq_in = placed.seq_in_frame
                seq_out = placed.seq_out_frame_exclusive

                repos.add_timeline_audio_clip(
                    ctx.db,
                    timeline_id=timeline_id,
                    asset_id=voice_asset["id"],
                    seq_in_frame=seq_in,
                    seq_out_frame_exclusive=seq_out,
                    asset_in_frame=0,
                    gain_percent=100,
                    fade_in_frames=3,
                    fade_out_frames=4,
                    mix_mode="replace_original",
                    ducking_percent=100,
                    label=f"reel-beat-{i}",
                )

                beats_result.append({
                    "voice_asset_id": voice_asset["id"],
                    "measured_frames": measured_frames,
                    "seq_in": seq_in,
                    "seq_out": seq_out,
                })

            # Resolve stable clip ids for whichever video clips exist now (whether every
            # beat ran, or the loop stopped early on cancellation) and fold them into the
            # per-beat results in the same append order (one video clip per completed
            # beat, on a timeline this job alone builds -- order-by-seq_in is beat order).
            # strict=True: a length mismatch here is an invariant break (one video clip
            # per completed beat, always), not a recoverable condition -- fail loudly.
            video_clips = [
                c for c in repos.list_timeline_clips(ctx.db, timeline_id)
                if c.get("role", "base") != "replace"
            ]
            for entry, row in zip(beats_result, video_clips, strict=True):
                entry["clip_id"] = row["id"]

            if cancelled:
                return {
                    "status": "cancelled",
                    "beats": beats_result,
                    "export_id": None,
                    "warnings": warnings,
                }

            n = len(video_clips)
            for idx, row in enumerate(video_clips):
                if idx == n - 1:
                    if final_fade_frames > 0:
                        repos.set_clip_transition(
                            ctx.db, clip_id=row["id"], kind="fade", frames=final_fade_frames
                        )
                elif crossfade_frames > 0:
                    repos.set_clip_transition(
                        ctx.db, clip_id=row["id"], kind="crossfade", frames=crossfade_frames
                    )

            export_id: str | None = None
            if bool(payload.get("render", True)):
                options: dict[str, Any] = {
                    "captions": True,
                    "caption_source": "voiceover",
                    "caption_preset": payload.get("caption_preset") or "wide",
                }
                if payload.get("ausgabe") == "reel":
                    # Hochkant gleich mit Reel-Optionen (sonst renderte der Job quer).
                    options.update(
                        vertical=True,
                        caption_preset="reels",
                        hook_text=payload.get("hook_text"),
                        kennzeichnung="auto",
                        disclosure_text=None,
                    )
                    if payload.get("max_bytes"):
                        options["max_bytes"] = int(payload["max_bytes"])
                exp = repos.create_export(
                    ctx.db,
                    project_id=timeline["project_id"],
                    timeline_id=timeline_id,
                    format="mp4",
                    options=options,
                )
                export_id = exp["id"]
                enqueue(
                    ctx.db,
                    queue=queue_for("export.render"),
                    kind="export.render",
                    payload={"export_id": export_id},
                    idempotency_key=f"render:{export_id}",
                )
        except Exception:
            # A mid-build failure must never leave a half-built collage on the timeline
            # (spec's Fehlerfälle: "Timeline bleibt im letzten konsistenten Zustand").
            # perform_undo pops exactly the checkpoint pushed above (nothing else pushes
            # one during this job) and restores clips/audio clips/transitions/scenes to
            # the empty pre-job snapshot -- the SAME machinery the /undo endpoint uses.
            # Synthetic voice assets already created stay (project-level, harmless), but
            # every timeline clip and VO clip from this run is gone. Re-raise so the job
            # still fails with the original, clear error.
            perform_undo(ctx.db, timeline_id)
            raise

    return {
        "beats": beats_result,
        "export_id": export_id,
        "warnings": warnings,
    }


def _object_bool(obj: object, name: str) -> bool:
    return bool(getattr(obj, name))


def _object_float(obj: object, name: str) -> float:
    return float(getattr(obj, name))


def _probe_dict(probe: object) -> dict[str, object]:
    reason = getattr(probe, "reason", None)
    return {
        "face_detected": _object_bool(probe, "face_detected"),
        "mouth_visible": _object_bool(probe, "mouth_visible"),
        "audio_present": _object_bool(probe, "audio_present"),
        "reason": reason if isinstance(reason, str) else None,
    }


def _quality_dict(quality: object) -> dict[str, object]:
    return {
        "sync_score": _object_float(quality, "sync_score"),
        "mouth_score": _object_float(quality, "mouth_score"),
        "temporal_score": _object_float(quality, "temporal_score"),
        "passed": _object_bool(quality, "passed"),
    }


def _quality_passes(quality: object, threshold: float) -> bool:
    scores = [
        _object_float(quality, "sync_score"),
        _object_float(quality, "mouth_score"),
        _object_float(quality, "temporal_score"),
    ]
    return _object_bool(quality, "passed") and min(scores) >= threshold


def handle_lipsync(ctx: JobContext) -> dict[str, Any]:
    """Consent- and license-gated lipsync job.

    Gate order is safety-critical: license and consent are checked before any
    temp media render or sidecar call.
    """
    payload = ctx.payload
    if payload.get("license_accepted") is not True:
        raise ValueError("ai.lipsync: license_accepted must be true")

    consent_id = str(payload.get("consent_id") or "")
    if not consent_id:
        raise ValueError("ai.lipsync: payload missing required key 'consent_id'")
    consent = repos.get_consent_record(ctx.db, consent_id)
    if consent is None:
        raise ValueError(f"ai.lipsync: consent record not found: {consent_id!r}")
    if consent.get("revoked_at"):
        raise ValueError(f"ai.lipsync: consent {consent_id!r} has been revoked")

    timeline_id = str(payload["timeline_id"])
    timeline = repos.get_timeline(ctx.db, timeline_id)
    if timeline is None:
        raise ValueError(f"ai.lipsync: timeline not found: {timeline_id!r}")
    project = repos.get_project(ctx.db, timeline["project_id"])
    if project is None:
        raise ValueError(f"ai.lipsync: project not found: {timeline['project_id']!r}")
    if consent["project_id"] != timeline["project_id"]:
        raise ValueError("ai.lipsync: consent does not belong to this timeline project")

    audio_asset_id = str(payload["audio_asset_id"])
    audio_asset = repos.get_asset(ctx.db, audio_asset_id)
    if audio_asset is None:
        raise ValueError(f"ai.lipsync: audio asset not found: {audio_asset_id!r}")
    if audio_asset["project_id"] != timeline["project_id"]:
        raise ValueError("ai.lipsync: audio asset does not belong to this timeline project")
    if audio_asset.get("type") != "audio" and not audio_asset.get("codec_audio"):
        raise ValueError("ai.lipsync: selected asset has no audio stream")

    rate_num = int(project["sequence_rate_num"])
    rate_den = int(project["sequence_rate_den"])
    seq_in = int(payload["seq_in_frame"])
    seq_out = int(payload["seq_out_frame_exclusive"])
    if seq_out <= seq_in:
        raise ValueError("ai.lipsync: seq_out_frame_exclusive must be greater than seq_in_frame")
    duration_frames = seq_out - seq_in

    if timeline.get("kind") == "sequence":
        base_rows = flatten_sequence(ctx.db, timeline["id"])
    else:
        base_rows = [
            row
            for row in repos.list_timeline_clips(ctx.db, timeline["id"])
            if row.get("role", "base") != "replace"
        ]
    driving_clips: list[tuple[Path, int, int]] = []
    for row in base_rows:
        row_seq_in = int(row["seq_in_frame"])
        row_seq_out = int(row["seq_out_frame_exclusive"])
        overlap_in = max(row_seq_in, seq_in)
        overlap_out = min(row_seq_out, seq_out)
        if overlap_in >= overlap_out:
            continue
        asset = repos.get_asset(ctx.db, row["asset_id"])
        if asset is None:
            raise ValueError(f"ai.lipsync: asset not found: {row['asset_id']!r}")
        src_in = int(row["src_in_frame"]) + (overlap_in - row_seq_in)
        src_out = int(row["src_in_frame"]) + (overlap_out - row_seq_in)
        driving_clips.append((Path(asset["source_path"]), src_in, src_out))
    if not driving_clips:
        raise ValueError(f"ai.lipsync: no base clips overlap range [{seq_in}, {seq_out})")

    workspace = Path(project["workspace_root"])
    tmp_dir = workspace / "tmp"
    synthetic_dir = workspace / "synthetic"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    driving_tmp = tmp_dir / f"{new_id()}.lipsync-driving.mp4"
    out_path = synthetic_dir / f"{new_id()}.lipsync.mp4"
    probe_payload: dict[str, object]
    quality_payload: dict[str, object]

    try:
        render_clips_mp4(driving_clips, driving_tmp, rate_num=rate_num, rate_den=rate_den)
        backend_config = _backend_config_from_runtime(
            ctx.db,
            payload.get("runtime_id"),
            payload.get("backend"),
            "lipsync",
        )
        backend = resolve_lipsync_backend(
            backend_config.name,
            base_url=backend_config.base_url,
        )
        if not backend.available():
            raise RuntimeError(f"ai.lipsync: lipsync backend '{backend.name}' is not installed")

        probe = backend.probe(video_path=driving_tmp, audio_path=Path(audio_asset["source_path"]))
        probe_payload = _probe_dict(probe)
        if not (
            probe_payload["face_detected"]
            and probe_payload["mouth_visible"]
            and probe_payload["audio_present"]
        ):
            reason = probe_payload["reason"] or "face/mouth/audio probe failed"
            raise ValueError(f"ai.lipsync: {reason}")

        quality = backend.lipsync(
            video_path=driving_tmp,
            audio_path=Path(audio_asset["source_path"]),
            out_path=out_path,
            fps_num=rate_num,
            fps_den=rate_den,
        )
        assert_or_fix_media_sync(
            out_path,
            expected_frames=duration_frames,
            rate_num=rate_num,
            rate_den=rate_den,
            require_video=True,
            require_audio=True,
            fix=True,
        )
        quality_payload = _quality_dict(quality)
        threshold = float(payload.get("quality_threshold", 0.6))
        if not _quality_passes(quality, threshold):
            raise ValueError(
                f"ai.lipsync: quality gate failed below threshold {threshold:.2f}"
            )
    except Exception:
        out_path.unlink(missing_ok=True)
        raise
    finally:
        driving_tmp.unlink(missing_ok=True)

    asset = repos.create_asset(
        ctx.db,
        project_id=timeline["project_id"],
        type="video",
        display_name=f"lipsync {seq_in}-{seq_out}",
        source_path=str(out_path),
        synthetic=True,
        ai_effect="lipsync",
    )
    write_ai_provenance_manifest(
        media_path=out_path,
        asset_id=asset["id"],
        project_id=timeline["project_id"],
        ai_effect="lipsync",
        source={
            "timeline_id": timeline["id"],
            "audio_asset_id": audio_asset_id,
            "consent_id": consent_id,
            "backend": payload.get("backend") or "stub",
            "quality_threshold": payload.get("quality_threshold", 0.6),
            "seq_in_frame": seq_in,
            "seq_out_frame_exclusive": seq_out,
            "probe": probe_payload,
            "quality": quality_payload,
        },
    )
    if repos.is_job_cancel_requested(ctx.db, ctx.job_id):
        return {"status": "cancelled", "reason": "undo"}
    repos.add_timeline_clip(
        ctx.db,
        timeline_id=timeline["id"],
        asset_id=asset["id"],
        src_in_frame=0,
        src_out_frame_exclusive=duration_frames,
        seq_in_frame=seq_in,
        seq_out_frame_exclusive=seq_out,
        lane=1,
        role="replace",
    )
    try:
        audit.record(
            ctx.db,
            audit.system_principal(),
            "ai.lipsync",
            entity_type="media_asset",
            entity_id=asset["id"],
            payload={
                "timeline_id": timeline["id"],
                "audio_asset_id": audio_asset_id,
                "consent_id": consent_id,
                "seq_in_frame": seq_in,
                "seq_out_frame_exclusive": seq_out,
            },
        )
    except Exception:
        _log.warning("audit.record failed for ai.lipsync; job result preserved", exc_info=True)

    return {
        "asset_id": asset["id"],
        "audio_asset_id": audio_asset_id,
        "consent_id": consent_id,
        "out_path": str(out_path),
        "seq_in_frame": seq_in,
        "seq_out_frame_exclusive": seq_out,
        "probe": probe_payload,
        "quality": quality_payload,
    }
