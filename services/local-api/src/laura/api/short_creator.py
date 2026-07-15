"""NL-agent short-creator endpoint (Iteration 8).

``POST /assets/{asset_id}/auto-short`` enqueues a ``short_creator.run`` job that builds a short
about *topic* from the asset via the multi-agent escalation ladder. Requires the optional
``autoshort`` extra; without it the endpoint returns 503. Poll ``/jobs/{id}`` for completion.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..auth import Principal, require_permission
from ..db import repos
from ..db.database import Database
from ..jobs.queues import queue_for
from ..jobs.runner import enqueue
from ..util import new_id

if TYPE_CHECKING:  # annotation only — never imported at runtime
    from ..short_creator.board import Board

logger = logging.getLogger(__name__)

router = APIRouter(tags=["short-creator"])


class AutoShortRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=2000)
    target_seconds: int = Field(default=60, gt=0, le=600)


class ProductionCreateRequest(BaseModel):
    task: str = Field(min_length=1, max_length=2000)
    target_seconds: int = Field(default=60, gt=0, le=600)


class ProductionMessageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


def _db(request: Request) -> Database:
    db: Database = request.app.state.db
    return db


def _autoshort_available() -> bool:
    """True when the optional ``autoshort`` extra (AutoGen) is importable."""
    try:
        import autogen_agentchat  # noqa: F401
    except ImportError:
        return False
    return True


def _require_autoshort() -> None:
    """Raise 503 unless the optional ``autoshort`` extra (AutoGen) is installed.

    Shared by every endpoint that enqueues an agent run (extracted once a 4th call site —
    the follow-up message endpoint — joined the original three).
    """
    if not _autoshort_available():
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The 'autoshort' extra is not installed. Run: uv sync --extra autoshort",
        )


def _get_asset_or_404(db: Database, asset_id: str) -> dict[str, Any]:
    """Return the asset row for *asset_id*, or raise 404 if it does not exist."""
    asset = repos.get_asset(db, asset_id)
    if asset is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "asset not found")
    return asset


def _open_board_or_404(db: Database, asset_id: str, session_id: str) -> Board:
    """Open *session_id*'s production board, or raise 404 if there isn't one yet.

    Covers both ``board_root_for``'s ``ValueError`` (the asset — or its project — no longer
    exists) and ``Board.open``'s ``FileNotFoundError`` (no board written for this session
    yet): from a caller's point of view both mean "nothing to read/append to here".
    """
    from ..short_creator.board import Board
    from ..short_creator.production_orchestrator import board_root_for

    try:
        root = board_root_for(db, asset_id, session_id)
        return Board.open(root)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "board missing") from exc


def _expected_scenes_for(db: Database, asset_id: str) -> list[int]:
    """Rough-cut scene numbers for *asset_id* (the reviews the board should cover).

    Mirrors ``production_orchestrator._expected_scene_numbers``'s logic on top of
    :func:`laura.short_creator.context.scene_transcripts` — kept as a local copy rather than
    importing that helper since its leading underscore marks it private to its own module.
    """
    from ..short_creator import context

    result = context.scene_transcripts(db, asset_id)
    if not result.get("ok"):
        return []
    return [int(s["scene_number"]) for s in result.get("scenes", [])]


@router.post("/assets/{asset_id}/auto-short", status_code=status.HTTP_202_ACCEPTED)
def auto_short(
    asset_id: str,
    body: AutoShortRequest,
    request: Request,
    principal: Annotated[Principal, Depends(require_permission("timeline:edit"))],
) -> dict[str, Any]:
    """Enqueue a ``short_creator.run`` job for *asset_id* (503 if the extra is missing)."""
    db = _db(request)
    _get_asset_or_404(db, asset_id)
    _require_autoshort()
    payload: dict[str, Any] = {
        "asset_id": asset_id,
        "topic": body.topic,
        "target_seconds": body.target_seconds,
    }
    # LLM runs are expensive + non-idempotent — do not auto-retry.
    job_id = enqueue(
        db,
        queue=queue_for("short_creator.run"),
        kind="short_creator.run",
        payload=payload,
        max_attempts=1,
    )
    return {"job_id": job_id}


@router.post("/assets/{asset_id}/auto-short/stream")
def auto_short_stream(
    asset_id: str,
    body: AutoShortRequest,
    request: Request,
    principal: Annotated[Principal, Depends(require_permission("timeline:edit"))],
) -> StreamingResponse:
    """Run the short-creator live and stream normalized NDJSON events (one JSON object per line).

    The run is bound to this connection (closing it stops the run) — that is the "watch it live"
    behavior. 404 (asset) is checked before 503 (missing extra), both before streaming starts.
    """
    db = _db(request)
    _get_asset_or_404(db, asset_id)
    _require_autoshort()
    from ..short_creator.providers import resolve_from_env
    from ..short_creator.stream import run_short_creator_stream

    config = resolve_from_env()
    topic, target_seconds = body.topic, body.target_seconds

    # Every event is ALSO appended to an NDJSON run log (flushed per line), so a run can be
    # debugged after the fact — the chat panel's copy dies with the window (live finding:
    # "paste me the chat ending" was the only way to diagnose a run).
    log_file: IO[str] | None = None
    log_dir = request.app.state.settings.workspace_root / "agent-runs"
    log_path = log_dir / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{asset_id[:8]}.ndjson"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("a", encoding="utf-8")
        logger.info("auto-short run log: %s", log_path)
    except OSError as exc:  # logging must never block the run
        logger.warning("auto-short run log unavailable (%s): %s", log_path, exc)

    def write_line(event: dict[str, Any]) -> bytes:
        line = json.dumps(event, ensure_ascii=False)
        if log_file is not None:
            try:
                log_file.write(line + "\n")
                log_file.flush()
            except OSError:
                pass  # never break the stream over a full disk
        return (line + "\n").encode("utf-8")

    async def events() -> AsyncIterator[bytes]:
        try:
            write_line(
                {
                    "type": "meta",
                    "asset_id": asset_id,
                    "topic": topic,
                    "target_seconds": target_seconds,
                    "provider": config.provider,
                    "agent_model": config.agent_model,
                }
            )
            try:
                async for event in run_short_creator_stream(
                    db, config, asset_id=asset_id, topic=topic, target_seconds=target_seconds
                ):
                    yield write_line(event)
            except Exception as exc:  # never leak a raw 500 mid-stream — emit a final error
                yield write_line({"type": "error", "message": str(exc)})
        except BaseException:  # client disconnect kills the run — make that visible in the log
            write_line({"type": "aborted", "reason": "client disconnected"})
            raise
        finally:
            if log_file is not None:
                log_file.close()

    return StreamingResponse(events(), media_type="application/x-ndjson")


# --- v2 production session endpoint (Slice 4) -------------------------------------------------


@router.post("/assets/{asset_id}/production", status_code=status.HTTP_202_ACCEPTED)
def create_production(
    asset_id: str,
    body: ProductionCreateRequest,
    request: Request,
    principal: Annotated[Principal, Depends(require_permission("timeline:edit"))],
) -> dict[str, Any]:
    """Create a v2 production session for *asset_id* and enqueue its ``production.run`` job.

    The session row is created before the job is enqueued: a session without a job is
    harmless (visible, just never progresses), while a job without a session row would
    reference an entity that doesn't exist. 503 if the 'autoshort' extra is missing.
    """
    db = _db(request)
    _get_asset_or_404(db, asset_id)
    _require_autoshort()
    session_id = new_id()
    created_utc = datetime.now(UTC).isoformat(timespec="seconds")
    repos.create_production_session(
        db, session_id=session_id, asset_id=asset_id, created_utc=created_utc
    )
    # LLM-driven production runs are expensive + non-idempotent — do not auto-retry.
    job_id = enqueue(
        db,
        queue=queue_for("production.run"),
        kind="production.run",
        payload={
            "asset_id": asset_id,
            "session_id": session_id,
            "task": body.task,
            "target_seconds": body.target_seconds,
        },
        max_attempts=1,
    )
    return {"session_id": session_id, "job_id": job_id}


# --- v2 production follow-up + status endpoints (Slice 4) -------------------------------------


@router.post("/production/{session_id}/message", status_code=status.HTTP_202_ACCEPTED)
def send_production_message(
    session_id: str,
    body: ProductionMessageRequest,
    request: Request,
    principal: Annotated[Principal, Depends(require_permission("timeline:edit"))],
) -> dict[str, Any]:
    """Enqueue a follow-up ``production.run`` job on top of *session_id*'s existing board.

    404 if the session is unknown, 503 if the 'autoshort' extra is missing, 404 if the session
    has no board yet (a follow-up assumes a prior production run — there is nothing to follow up
    on otherwise). ``task``/``target_seconds`` are read back from the board's own meta (fixed at
    session creation, Task 5) rather than accepted again here — the follow-up only supplies the
    new ``message`` text.
    """
    db = _db(request)
    session = repos.get_production_session(db, session_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "session not found")
    _require_autoshort()
    asset_id = str(session["asset_id"])
    board = _open_board_or_404(db, asset_id, session_id)
    meta = board.meta()
    # LLM-driven production runs are expensive + non-idempotent — do not auto-retry.
    job_id = enqueue(
        db,
        queue=queue_for("production.run"),
        kind="production.run",
        payload={
            "asset_id": asset_id,
            "session_id": session_id,
            "task": meta.task,
            "target_seconds": int(meta.target_seconds),
            "message": body.text,
        },
        max_attempts=1,
    )
    return {"session_id": session_id, "job_id": job_id}


@router.get("/production/{session_id}")
def get_production_status(
    session_id: str,
    request: Request,
    principal: Annotated[Principal, Depends(require_permission("read"))],
) -> dict[str, Any]:
    """Read-only board status for *session_id*: ``board.status()`` plus the resume point.

    404 if the session is unknown, 404 if it has no board yet. Never enqueues anything and
    never requires the 'autoshort' extra — this is a pure read of what is already on disk.
    A present contact_sheet artifact additionally carries its ``png_path``/``labeled``/``tiles``
    inside the artifacts block (next to the chain-standard version fields), so a client can
    show the checkpoint without a second lookup; the image bytes come from
    ``GET /production/{session_id}/contact-sheet``.
    """
    from ..short_creator.board_models import ContactSheet

    db = _db(request)
    session = repos.get_production_session(db, session_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "session not found")
    asset_id = str(session["asset_id"])
    board = _open_board_or_404(db, asset_id, session_id)
    expected_scenes = _expected_scenes_for(db, asset_id)
    result = board.status()
    result["resume_point"] = board.resume_point(expected_scenes)
    sheet = board.load("contact_sheet")
    if isinstance(sheet, ContactSheet):
        result["artifacts"]["contact_sheet"].update(
            png_path=sheet.png_path,
            labeled=sheet.labeled,
            tiles=[t.model_dump() for t in sheet.tiles],
        )
    return result


@router.get("/production/{session_id}/contact-sheet")
def get_production_contact_sheet(
    session_id: str,
    request: Request,
    principal: Annotated[Principal, Depends(require_permission("read"))],
) -> FileResponse:
    """Serve the session's current contact-sheet PNG (the visual pre-render checkpoint).

    DB-lookup-before-fs guard (mirrors ``GET /shots/{shot_id}/thumbnail``): *session_id* only
    ever resolves through the sessions table and the board's own artifact record — the served
    path is the one ``save_contact_sheet`` wrote into the board, never one derived from client
    input. 404 for an unknown session, a board without a contact_sheet yet, or a png missing
    on disk.
    """
    from ..short_creator.board_models import ContactSheet

    db = _db(request)
    session = repos.get_production_session(db, session_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "session not found")
    board = _open_board_or_404(db, str(session["asset_id"]), session_id)
    sheet = board.load("contact_sheet")
    if not isinstance(sheet, ContactSheet):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no contact sheet on the board yet")
    path = Path(sheet.png_path)
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "contact sheet missing on disk")
    return FileResponse(path, media_type="image/png")
