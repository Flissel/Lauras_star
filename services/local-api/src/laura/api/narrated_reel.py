"""Narrated-reel collage-builder endpoint (spec §6): a beat list -> a finished timeline.

POST /projects/{project_id}/narrated-reel enqueues the ``ai.narrated_reel`` job, which
does all the heavy lifting (per-beat natural-length voiceover synthesis, clip placement,
transitions, optional render). This module only does the cheap synchronous validation
(project/asset existence, project membership, type/online, src_in bound) and creates the
target timeline before handing off to the job.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ..db import repos
from ..db.database import Database
from ..jobs.keys import idempotency_key_for
from ..jobs.queues import queue_for
from ..jobs.runner import enqueue
from .models import NarratedReelAccepted, NarratedReelRequest
from .security import require_token

router = APIRouter(tags=["narrated-reel"], dependencies=[Depends(require_token)])

# Mirrors jobs.runner.enqueue's own internal dedup rule exactly: a job in one of these
# statuses is still "live" (or already finished successfully) and safe to hand back as-is.
# failed/cancelled is deliberately excluded -- enqueue() drops a stale failed/cancelled row
# and starts a fresh one under the same key, so this endpoint's pre-check must agree.
_REUSABLE_JOB_STATUSES = ("queued", "leased", "running", "succeeded")


def _db(request: Request) -> Database:
    db: Database = request.app.state.db
    return db


def _validate_runtime(db: Database, runtime_id: str | None, effect: str) -> None:
    if runtime_id is None:
        return
    runtime = repos.get_ai_runtime(db, runtime_id)
    if runtime is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "runtime not found")
    if runtime["effect"] != effect:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"runtime effect must be {effect}",
        )
    if not runtime["enabled"]:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "runtime is disabled")


@router.post(
    "/projects/{project_id}/narrated-reel",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=NarratedReelAccepted,
)
def create_narrated_reel(
    project_id: str,
    body: NarratedReelRequest,
    request: Request,
) -> NarratedReelAccepted:
    """Validate the beat list, create the target timeline, and enqueue ``ai.narrated_reel``."""
    db = _db(request)
    project = repos.get_project(db, project_id)
    if project is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "project not found")

    for i, beat in enumerate(body.beats):
        asset = repos.get_asset(db, beat.asset_id)
        if asset is None:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, f"beat {i}: asset not found"
            )
        if asset["project_id"] != project_id:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"beat {i}: asset belongs to another project",
            )
        if asset["type"] != "video":
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, f"beat {i}: asset is not a video"
            )
        if not asset["online"]:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, f"beat {i}: asset is offline"
            )
        duration_frames = asset.get("duration_frames")
        if duration_frames is None or beat.src_in_frame >= duration_frames:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"beat {i}: src_in_frame must be less than the asset's duration_frames",
            )

    _validate_runtime(db, body.runtime_id, "voice")

    # The dedup key is computed WITHOUT a timeline_id -- there is no timeline yet at this
    # point, and jobs.keys.idempotency_key_for's ai.narrated_reel branch never reads
    # timeline_id anyway (see its docstring), so this dict hashes identically to the full
    # payload built below once timeline_id is added. A retried identical request must be
    # caught HERE, before create_timeline runs: enqueue()'s own internal dedup only fires
    # after the (would-be duplicate) timeline already exists, which is too late.
    key_payload: dict[str, Any] = {
        "project_id": project_id,
        "beats": [b.model_dump() for b in body.beats],
        "crossfade_frames": body.crossfade_frames,
        "final_fade_frames": body.final_fade_frames,
        "backend": body.backend,
        "voice_id": body.voice_id,
        "language": body.language,
        "runtime_id": body.runtime_id,
        "render": body.render,
        "caption_preset": body.caption_preset,
    }
    # Nur bei Hochkant-Ausgabe: alte Anfragen behalten ihren Idempotenz-Schluessel.
    if body.ausgabe == "reel":
        key_payload.update(ausgabe="reel", hook_text=body.hook_text, max_bytes=body.max_bytes)
    idempotency_key = idempotency_key_for("ai.narrated_reel", key_payload)
    # Accepted race: two simultaneous identical POSTs both read "no existing job" here and
    # both proceed to create_timeline below, orphaning one timeline. Single-user local app —
    # accepted rather than adding cross-request locking for this narrow window.
    if idempotency_key is not None:
        existing = repos.get_job_by_idempotency_key(db, idempotency_key)
        if existing is not None and existing["status"] in _REUSABLE_JOB_STATUSES:
            existing_payload = json.loads(existing["payload_json"])
            return NarratedReelAccepted(
                timeline_id=existing_payload["timeline_id"], job_id=existing["id"]
            )

    name = body.name or f"Narrated Reel {datetime.now(UTC):%Y-%m-%d}"
    timeline = repos.create_timeline(db, project_id=project_id, name=name, kind="rough_cut")

    payload: dict[str, Any] = {"timeline_id": timeline["id"], **key_payload}
    job_id = enqueue(
        db,
        queue=queue_for("ai.narrated_reel", default="ai"),
        kind="ai.narrated_reel",
        payload=payload,
        max_attempts=1,
        idempotency_key=idempotency_key,
    )
    return NarratedReelAccepted(timeline_id=timeline["id"], job_id=job_id)
