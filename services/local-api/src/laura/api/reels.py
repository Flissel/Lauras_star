"""Reel-render endpoint — mp4 export with social-reel options + render job enqueue."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ..db import repos
from ..db.database import Database
from ..jobs.queues import queue_for
from ..jobs.runner import enqueue
from ..ledger import get_ledger_store, mint_short_run
from ..ledger.recipe import RECIPE_EXCLUDED_KEYS
from .models import ReelRenderRequest
from .quality_gate import evaluate_quality_gate
from .security import require_token

_log = logging.getLogger(__name__)

router = APIRouter(tags=["reels"], dependencies=[Depends(require_token)])


def _db(request: Request) -> Database:
    db: Database = request.app.state.db
    return db


@router.post("/timelines/{timeline_id}/render-reel", status_code=status.HTTP_202_ACCEPTED)
def render_reel(
    timeline_id: str, body: ReelRenderRequest, request: Request
) -> dict[str, str]:
    """Enqueue an MP4 reel render job for a timeline and return the export record id.

    Mirrors ``render_timeline`` but carries social-reel options (vertical framing,
    hook text, disclosure text) forwarded by ``handle_render`` to the renderer.
    """
    db = _db(request)
    tl = repos.get_timeline(db, timeline_id)
    if tl is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "timeline not found")
    gate = evaluate_quality_gate(db, timeline_id, body.min_quality)

    # Build the options dict WITHOUT short_run_id first — it becomes the recipe so that the
    # content-addressed short_id is independent of the run row id (avoids circular derivation).
    options: dict[str, object] = {
        "vertical": body.vertical,
        "hook_text": body.hook_text,
        "disclosure_text": body.disclosure_text,
        "captions": body.captions,
        "caption_preset": body.caption_preset,
        "caption_mode": body.caption_mode,
        "caption_position": body.caption_position,
        "caption_fontsize": body.caption_fontsize,
        "caption_safe_margin": body.caption_safe_margin,
        "max_duration_seconds": body.max_duration_seconds,
        **gate,
    }
    # Nur gesetzt, wenn abweichend: alte Aufrufer behalten ihre Optionen (und damit ihren
    # Rezept-Hash) byte-gleich.
    if body.logo_asset_id:
        logo = repos.get_asset(db, body.logo_asset_id)
        if logo is None or logo["project_id"] != tl["project_id"] or logo["type"] != "image":
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                "logo_asset_id muss ein Bild-Asset dieses Projekts sein")
        options.update(logo_asset_id=body.logo_asset_id, logo_position=body.logo_position,
                       logo_breite_prozent=body.logo_breite_prozent)
    if body.caption_source is not None:
        options["caption_source"] = body.caption_source
    if body.max_bytes is not None:
        options["max_bytes"] = body.max_bytes
    if body.kennzeichnung == "auto":
        options["kennzeichnung"] = "auto"
        options["disclosure_text"] = None

    # Mint a short_run so the ledger tracks this build attempt from the moment it is enqueued.
    # Resolve the source asset for content-addressing (may be None for non-asset timelines).
    asset_id = tl.get("created_from")
    asset = repos.get_asset(db, str(asset_id)) if asset_id else None
    input_sha256: str | None = asset.get("sha256") if asset else None
    try:
        store = get_ledger_store(db)
        # Exclude runtime-observation keys so the recipe hash is stable across quality states.
        recipe = {k: v for k, v in options.items() if k not in RECIPE_EXCLUDED_KEYS}
        run = mint_short_run(store, recipe=recipe, input_sha256=input_sha256)
        options["short_run_id"] = run["id"]
    except Exception:  # noqa: BLE001
        # A ledger hiccup must NEVER block the render enqueue.
        _log.exception("mint_short_run failed for timeline %s (ledger skipped)", timeline_id)

    exp = repos.create_export(
        db,
        project_id=tl["project_id"],
        timeline_id=timeline_id,
        format="mp4",
        options=options,
    )
    job_id = enqueue(
        db,
        queue=queue_for("export.render"),
        kind="export.render",
        payload={"export_id": exp["id"]},
        idempotency_key=f"render:{exp['id']}",
    )
    return {"export_id": exp["id"], "job_id": job_id}
