"""Deterministic idempotency-key builder for all job kinds.

Single source of truth for key construction.  The produced strings are
**byte-identical** to what each enqueue site built inline in Task 1 — the
refactor is purely structural, no semantic change.

Usage::

    from laura.jobs.keys import idempotency_key_for

    key = idempotency_key_for(kind, payload)   # str | None
    enqueue(db, ..., idempotency_key=key)

Kinds covered
-------------
``export.render``
    ``render:{export_id}``

``ai.lipsync``
    ``ai.lipsync:{sha256( timeline_id | seq_in_frame | seq_out_frame_exclusive
                          | audio_asset_id | consent_id )}``

``ai.reenact``
    ``ai.reenact:{sha256( timeline_id | seq_in_frame | seq_out_frame_exclusive
                          | portrait_asset_id | consent_id )}``

``ai.voiceover``
    ``ai.voiceover:{sha256( timeline_id | seq_in_frame | seq_out_frame_exclusive
                            | segment_id | text* | voice_id | mix_mode
                            | ducking_percent | fit | pad_frames )}``
    *text is included only when segment_id is absent/None (matches Task-1 logic).
    fit/pad_frames were added in the natural-length-fit task (2026-08-21 narrated-reel
    design §3): they change the rendered clip span, so two otherwise-identical requests
    that differ only in fit mode must not dedupe to the same job.

``ai.narrated_reel``
    ``ai.narrated_reel:{sha256( project_id | beats | crossfade_frames |
                                final_fade_frames | backend | voice_id | language |
                                runtime_id | render | caption_preset )}``
    ``timeline_id`` is DELIBERATELY EXCLUDED (not merely absent-tolerant — the branch
    below only ever reads the named fields, so a payload that happens to carry
    ``timeline_id`` hashes identically to one that doesn't). The endpoint
    (api/narrated_reel.py) must look up an existing job by this key BEFORE creating a
    timeline: since ``timeline_id`` is freshly minted per request, including it in the
    hash would make every request's key unique and the dedup path unreachable. Once a
    reusable job is found, the endpoint returns that job's own ``timeline_id`` (read
    back from its stored payload) instead of creating a second one.

All other kinds return ``None`` — no dedup key.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _sha256_of(parts: dict[str, object]) -> str:
    """SHA-256 hex digest of a JSON-serialised dict with sorted keys."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def idempotency_key_for(kind: str, payload: dict[str, Any]) -> str | None:
    """Return the deterministic idempotency key for *kind* + *payload*, or None.

    The produced strings are byte-identical to what the individual enqueue sites
    built inline before this refactor.  Do NOT change the key strings without a
    matching migration / cache-bust.
    """
    if kind == "export.render":
        export_id = payload.get("export_id")
        if export_id is None:
            return None
        return f"render:{export_id}"

    if kind == "ai.lipsync":
        parts: dict[str, object] = {
            "timeline_id": payload.get("timeline_id"),
            "seq_in_frame": payload.get("seq_in_frame"),
            "seq_out_frame_exclusive": payload.get("seq_out_frame_exclusive"),
            "audio_asset_id": payload.get("audio_asset_id"),
            "consent_id": payload.get("consent_id"),
        }
        return f"ai.lipsync:{_sha256_of(parts)}"

    if kind == "ai.reenact":
        parts = {
            "timeline_id": payload.get("timeline_id"),
            "seq_in_frame": payload.get("seq_in_frame"),
            "seq_out_frame_exclusive": payload.get("seq_out_frame_exclusive"),
            "portrait_asset_id": payload.get("portrait_asset_id"),
            "consent_id": payload.get("consent_id"),
        }
        return f"ai.reenact:{_sha256_of(parts)}"

    if kind == "ai.voiceover":
        segment_id = payload.get("segment_id")
        parts = {
            "timeline_id": payload.get("timeline_id"),
            "seq_in_frame": payload.get("seq_in_frame"),
            "seq_out_frame_exclusive": payload.get("seq_out_frame_exclusive"),
            "segment_id": segment_id,
            # text is included only when there is no segment_id — mirrors Task-1 logic:
            # body.text if body.segment_id is None else None
            "text": payload.get("text") if segment_id is None else None,
            "voice_id": payload.get("voice_id"),
            "mix_mode": payload.get("mix_mode"),
            "ducking_percent": payload.get("ducking_percent"),
            "fit": payload.get("fit"),
            "pad_frames": payload.get("pad_frames"),
        }
        return f"ai.voiceover:{_sha256_of(parts)}"

    if kind == "ai.narrated_reel":
        # timeline_id is NEVER read here, even if present in payload -- see the module
        # docstring on why this must stay reachable across the create-timeline-first flow.
        parts = {
            "project_id": payload.get("project_id"),
            "beats": payload.get("beats"),
            "crossfade_frames": payload.get("crossfade_frames"),
            "final_fade_frames": payload.get("final_fade_frames"),
            "backend": payload.get("backend"),
            "voice_id": payload.get("voice_id"),
            "language": payload.get("language"),
            "runtime_id": payload.get("runtime_id"),
            "render": payload.get("render"),
            "caption_preset": payload.get("caption_preset"),
        }
        # Hochkant-Ausgabe (17.10.2026): nur dann im Schluessel, damit alte Schluessel gleich
        # bleiben - und ein Reel nicht den gleichlautenden Quer-Auftrag wiederfindet.
        if payload.get("ausgabe") == "reel":
            parts.update(
                ausgabe="reel",
                hook_text=payload.get("hook_text"),
                max_bytes=payload.get("max_bytes"),
            )
        return f"ai.narrated_reel:{_sha256_of(parts)}"

    return None
