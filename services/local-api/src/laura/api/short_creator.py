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

# Pure-pydantic leaf: safe at runtime even without the optional 'autoshort' extra, unlike
# short_creator.board below.
from ..short_creator.board_models import Format
from ..short_creator.discovery import search_material
from ..short_creator.providers import config_warnings, resolve_from_env
from ..short_creator.scout import ScoutDecision, run_scout
from ..util import new_id

if TYPE_CHECKING:  # annotation only — never imported at runtime
    from ..short_creator.board import Board

logger = logging.getLogger(__name__)

router = APIRouter(tags=["short-creator"])


class AutoShortRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=2000)
    target_seconds: int = Field(default=60, gt=0, le=600)
    format: Format = "insta"
    language: str = Field(default="German", min_length=2, max_length=40)


class ProductionCreateRequest(BaseModel):
    task: str = Field(min_length=1, max_length=2000)
    target_seconds: int = Field(default=60, gt=0, le=600)
    # Delivery format picks the canvas: "insta" 9:16 reel, "x" native 16:9, "linkedin" 1:1.
    format: Format = "insta"
    # The script's language, named as it should be written ("German", "English").
    language: str = Field(default="German", min_length=2, max_length=40)


class ProductionMessageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class ProductionRevertRequest(BaseModel):
    artifact: str
    version: int


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


def _require_usable_agent_config() -> None:
    """Raise 503 unless the configured agent provider could actually be reached.

    Live incident: a run was started against ``openai-compat`` with no ``LAURA_AGENT_API_KEY``.
    It was enqueued, created a board, spent both escalation stages and came back "Connection
    error." — then looked alive for 55 minutes. The extra was checked; the credential was not.
    """
    from laura.short_creator.providers import config_problems, resolve_from_env

    problems = config_problems(resolve_from_env())
    if problems:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "the agent provider is not usable: " + "; ".join(problems),
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


def _restored_from_job(job: dict[str, Any]) -> list[str]:
    """The artifact names a resume restored from the provenance chain, read off *job*'s result.

    ``run_production`` always writes a ``restored`` key into its result dict (empty when nothing
    was restored), but this is a pure status read: a job with no result yet (still queued), an
    unparseable ``result_json``, or a result missing/mistyping the key must degrade to ``[]``
    rather than raise or 500 the whole status endpoint.
    """
    raw = job.get("result_json")
    if not raw:
        return []
    try:
        result = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(result, dict):
        return []
    restored = result.get("restored")
    if not isinstance(restored, list):
        return []
    return [item for item in restored if isinstance(item, str)]


def _job_view(job: dict[str, Any] | None) -> dict[str, Any] | None:
    """The liveness projection of *job* shared by ``GET /production/{sid}`` and the revert
    endpoint — ``None`` when there is no job on the session yet."""
    if job is None:
        return None
    return {
        "id": job["id"],
        "status": job["status"],
        "attempt": job["attempt"],
        "updated_at": job["updated_at"],
        "lease_expires_at": job["lease_expires_at"],
        "finished_at": job["finished_at"],
        "restored": _restored_from_job(job),
    }


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
    _require_usable_agent_config()
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
    _require_usable_agent_config()
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


def _create_production_session(
    db: Database,
    asset_id: str,
    *,
    task: str,
    target_seconds: float,
    format: str,
    language: str,
) -> tuple[str, str]:
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
            "task": task,
            "target_seconds": target_seconds,
            "format": format,
            "language": language,
        },
        max_attempts=1,
    )
    repos.set_production_session_job(db, session_id, job_id)
    return session_id, job_id


@router.post("/projects/{project_id}/auto-short", status_code=status.HTTP_202_ACCEPTED)
def create_project_auto_short(
    project_id: str,
    body: AutoShortRequest,
    request: Request,
    principal: Annotated[Principal, Depends(require_permission("timeline:edit"))],
) -> dict[str, Any]:
    """Scout project material for *topic* and enqueue a production session for the choice."""
    db = _db(request)
    if repos.get_project(db, project_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "project not found")
    _require_autoshort()
    _require_usable_agent_config()
    material = search_material(db, project_id, body.topic)
    if not material["ranking"]:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "reason": "no material found for topic",
                "skipped": material["skipped"],
                "source": material["source"],
            },
        )
    decision: ScoutDecision = run_scout(
        db,
        resolve_from_env(),
        project_id=project_id,
        topic=body.topic,
        material=material,
    )
    asset = repos.get_asset(db, decision["asset_id"])
    assert asset is not None
    chosen_material = next(
        entry for entry in material["ranking"] if entry["asset_id"] == decision["asset_id"]
    )
    snippets = [str(hit["snippet"]) for hit in chosen_material["scene_hits"]]
    task = (
        f"{body.topic}\n\nMaterial scout: use asset '{asset['display_name']}'. "
        f"Focus on scenes {', '.join(map(str, decision['scene_numbers']))} — "
        f"transcript hits: {'; '.join(snippets)}. Scout rationale: {decision['rationale']}"
    )
    session_id, job_id = _create_production_session(
        db,
        decision["asset_id"],
        task=task,
        target_seconds=body.target_seconds,
        format=body.format,
        language=body.language,
    )
    return {
        "session_id": session_id,
        "job_id": job_id,
        "asset_id": decision["asset_id"],
        "scene_numbers": decision["scene_numbers"],
        "rationale": decision["rationale"],
        "fallback": decision["fallback"],
        "ranking": material["ranking"],
        "warnings": config_warnings(resolve_from_env()),
    }


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
    _require_usable_agent_config()
    session_id, job_id = _create_production_session(
        db,
        asset_id,
        task=body.task,
        target_seconds=body.target_seconds,
        format=body.format,
        language=body.language,
    )

    return {
        "session_id": session_id,
        "job_id": job_id,
        "warnings": config_warnings(resolve_from_env()),
    }


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
    _require_usable_agent_config()
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
    repos.set_production_session_job(db, session_id, job_id)
    from ..short_creator.providers import config_warnings, resolve_from_env

    return {
        "session_id": session_id,
        "job_id": job_id,
        "warnings": config_warnings(resolve_from_env()),
    }


def _production_status_payload(
    db: Database,
    *,
    asset_id: str,
    board: Board,
    job_view: dict[str, Any] | None,
) -> dict[str, Any]:
    """The enriched board-status payload shared by GET /production/{sid} and the revert
    endpoint — extracted so the two responses can never drift apart."""
    from ..short_creator.board_models import ContactSheet

    result = board.status()
    result["resume_point"] = board.resume_point(_expected_scenes_for(db, asset_id))
    result["job"] = job_view
    result["board_ready"] = True
    sheet = board.load("contact_sheet")
    if isinstance(sheet, ContactSheet):
        result["artifacts"]["contact_sheet"].update(
            png_path=sheet.png_path,
            labeled=sheet.labeled,
            tiles=[t.model_dump() for t in sheet.tiles],
        )
    return result


@router.get("/production/{session_id}")
def get_production_status(
    session_id: str,
    request: Request,
    principal: Annotated[Principal, Depends(require_permission("read"))],
) -> dict[str, Any]:
    """Read-only status for *session_id*: the running job's liveness, plus the board when ready.

    404 only if the session is unknown. A session whose board does not exist yet — queued, or
    died before building one — returns ``{"job": ..., "board_ready": false}`` rather than 404,
    because that dead-before-a-board run is exactly the case liveness has to surface. Never
    enqueues anything and never requires the 'autoshort' extra — a pure read of what is on disk.
    A present contact_sheet artifact additionally carries its ``png_path``/``labeled``/``tiles``
    inside the artifacts block (next to the chain-standard version fields), so a client can
    show the checkpoint without a second lookup; the image bytes come from
    ``GET /production/{session_id}/contact-sheet``.
    """
    from ..short_creator.board import Board
    from ..short_creator.production_orchestrator import board_root_for

    db = _db(request)
    session = repos.get_production_session(db, session_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "session not found")
    asset_id = str(session["asset_id"])

    # Liveness: the one field the incident needed and did not have. The job is the authority on
    # whether the run is alive — a hanging run is a "running" job with an expired lease, a dead
    # run is a "failed" job — and it is looked up BEFORE the board, because the very failure
    # this closes is a run that died before a readable board existed. The board alone kept
    # reading "active" for 55 minutes; requiring it to answer would hide exactly the dead run.
    job_id = session.get("latest_job_id")
    job = repos.get_job(db, str(job_id)) if job_id else None
    job_view = _job_view(job)

    try:
        board = Board.open(board_root_for(db, asset_id, session_id))
    except (ValueError, FileNotFoundError):
        # No board yet — the run is queued, or it died before building one. That is a state to
        # report, not a 404: the job view carries the answer.
        return {"session_id": session_id, "job": job_view, "board_ready": False}

    return _production_status_payload(db, asset_id=asset_id, board=board, job_view=job_view)


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


@router.post("/production/{session_id}/revert")
def revert_production_artifact(
    session_id: str,
    body: ProductionRevertRequest,
    request: Request,
    principal: Annotated[Principal, Depends(require_permission("timeline:edit"))],
) -> dict[str, Any]:
    """Revert one board artifact to an archived version and heal the suffix — synchronously,
    no job and no agent turn. Mirrors the revert_artifact tool's validation; then
    ``board.revert`` + ``restore_coherent_suffix``. 409 while a job is queued/running (a
    revert under a live team run would race it). Returns the same enriched status payload
    as ``GET /production/{sid}`` so the UI updates without a second fetch.

    Accepted TOCTOU: the job-status check above races an in-flight enqueue by design — the
    worst case is a benign reorder (every run reads the board fresh at its own start), which
    is fine for a local single-user tool.
    """
    from ..short_creator.board import Board, downstream_of
    from ..short_creator.production_orchestrator import board_root_for

    db = _db(request)
    session = repos.get_production_session(db, session_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "session not found")
    job_id = session.get("latest_job_id")
    job = repos.get_job(db, str(job_id)) if job_id else None
    if job is not None and str(job["status"]) in ("queued", "running"):
        raise HTTPException(
            status.HTTP_409_CONFLICT, "run in progress — revert would race the team"
        )
    asset_id = str(session["asset_id"])
    try:
        board = Board.open(board_root_for(db, asset_id, session_id))
    except (ValueError, FileNotFoundError):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "session has no board") from None

    valid_names = downstream_of("scene_reviews")
    if body.artifact not in valid_names:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"unknown artifact '{body.artifact}'; valid: {', '.join(valid_names)}",
        )
    invalidated = [d for d in downstream_of(body.artifact) if board.load(d) is not None]
    try:
        board.revert(body.artifact, body.version)
    except FileNotFoundError:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"no archived {body.artifact} v{body.version}",
        ) from None
    restored = board.restore_coherent_suffix()

    job_view = _job_view(job)
    return {
        "ok": True,
        "artifact": body.artifact,
        "version": body.version,
        "invalidated": invalidated,
        "restored": restored,
        "status": _production_status_payload(
            db, asset_id=asset_id, board=board, job_view=job_view
        ),
    }
