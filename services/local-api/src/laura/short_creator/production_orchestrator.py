"""v2 production run entrypoint: Production Board lifecycle + resume-aware task contract.

:func:`run_production` is the v2 counterpart of :func:`orchestrator.run_short_creator`: instead
of a single flat task prompt, it opens (or creates) a session's :class:`board.Board` first, asks
:func:`build_production_task` to describe the CURRENT board state as part of the task text (so a
resumed run does not redo finished work), then runs the fixed five-agent production team
(:mod:`production_agents`) through a small magentic-only A/B escalation ladder. v2 has no
GraphFlow fallback, so unlike v1's ``orchestrator._run_stage`` a hard failure at Stage A goes
straight to Stage B instead of trying GraphFlow first at Stage A.

Team execution stays injectable (``execute: ExecuteFn | None``) for the same reason as v1: the
ladder policy is fully testable without an LLM (:func:`orchestrator._safe_execute` guarantees a
raising ``execute`` still becomes a clean ``hard_fail`` outcome, so ``run_production`` itself
never raises). The default executor lazily builds the real
:func:`production_agents.build_production_team` and runs it with ``asyncio.run`` (mirrors
:func:`orchestrator._default_execute`); it is board-bound (the team needs ``board``/``asset_id``/
``deps``), so it is built as a closure rather than a bare module-level function, and only imports
``asyncio`` inside its own body — no autogen (nor anything that transitively needs it) is
imported at this module's top level.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..db import repos
from ..db.database import Database
from . import context
from .board import Board
from .board_models import BoardMeta, QaReport, RenderReport
from .orchestrator import ExecuteFn, StageOutcome, _safe_execute
from .production_agents import build_production_team
from .production_tools import ProductionDeps
from .providers import AgentConfig, Stage

logger = logging.getLogger(__name__)


def board_root_for(db: Database, asset_id: str, session_id: str) -> Path:
    """``<project.workspace_root>/agent-runs/<session_id>/board`` for *asset_id*'s project.

    Pure path construction: does not touch the filesystem and does not require a board to exist
    yet at the returned path.
    """
    asset = repos.get_asset(db, asset_id)
    if asset is None:
        raise ValueError(f"asset not found: {asset_id}")
    project = repos.get_project(db, str(asset["project_id"]))
    if project is None:
        raise ValueError(f"project not found for asset: {asset_id}")
    return Path(str(project["workspace_root"])) / "agent-runs" / session_id / "board"


def _expected_scene_numbers(db: Database, asset_id: str) -> list[int]:
    """Scene numbers this asset's rough cut has (the reviews the board should eventually cover).

    Reimplements ``production_tools._expected_scenes``' logic locally on top of
    :func:`context.scene_transcripts` rather than importing that private helper, keeping this
    module's dependency on :mod:`production_tools` limited to :class:`ProductionDeps`.
    """
    result = context.scene_transcripts(db, asset_id)
    if not result.get("ok"):
        return []
    return [int(s["scene_number"]) for s in result.get("scenes", [])]


def build_production_task(
    db: Database,
    board: Board,
    *,
    asset_id: str,
    task: str,
    target_seconds: int,
    message: str | None = None,
) -> str:
    """The task text handed to the magentic production team: a resume-aware contract.

    Six fixed, deterministically-ordered elements, plus a seventh that only appears when this
    call is a follow-up on top of an already-produced board:

    1. goal + format + target length;
    2. the FIXED viral arc as a structural contract;
    3. the board's current resume status (reviews vs. expected scenes, per-artifact DONE/pending
       + version, the resume point, and an explicit "do not redo" instruction). When ``message``
       is set, any artifact that has archived versions also lists them (``[archived: vN, ...]``)
       so the team can name a concrete ``(name, version)`` pair back to ``revert_artifact``;
    4. the mandatory stage order (reviews -> storyline -> script -> voice+cutlist ->
       contact sheet -> render -> qa) plus the contact-sheet checkpoint as a known pattern:
       stopping at the Kontaktbogen or rendering later is steered purely by follow-up
       messages against the normal resume flow - no extra session state;
    5. the German-script language rule plus the coding-agent's ``voice_fits`` charter;
    6. the QA revision-round limit (one revise round, then ship with findings as warnings);
    7. only when ``message`` is set: the user's follow-up request text (capped at 2000 chars)
       plus instructions for interpreting it against the board status above - going back to an
       earlier version is a ``revert_artifact`` call using the ``archived_versions`` listed in
       section 3, a content change is a re-save of the affected artifact (the highest affected
       one wins when the request touches more than one) with everything downstream left to
       rebuild through the normal pipeline, and an intact upstream artifact the request does not
       mention must never be redone.
    """
    meta = board.meta()
    expected_scenes = _expected_scene_numbers(db, asset_id)
    status = board.status()
    resume_point = board.resume_point(expected_scenes)
    reviewed = status["scene_reviews"]["count"]
    show_archived = bool(message)

    def _artifact_line(name: str, info: dict[str, Any]) -> str:
        base = f"DONE (v{info['version']})" if info["version"] is not None else "pending"
        archived = info["archived_versions"]
        if show_archived and archived:
            versions = ", ".join(f"v{v}" for v in archived)
            return f"  - {name}: {base} [archived: {versions}]"
        return f"  - {name}: {base}"

    artifact_lines = "\n".join(
        _artifact_line(name, info) for name, info in status["artifacts"].items()
    )

    follow_up = ""
    if message:
        follow_up = (
            "\n"
            "7) USER FOLLOW-UP REQUEST:\n"
            f"   {message[:2000]}\n"
            "   Interpret this request against the BOARD STATUS above. Going back to an "
            "earlier version: coding_agent calls revert_artifact(name, version), using the "
            "archived_versions listed per artifact in the board status above. A content "
            "change: re-save the affected artifact - if the request touches more than one "
            "artifact, the highest affected artifact wins - and let everything downstream "
            "rebuild through the normal pipeline. NEVER redo an intact upstream artifact the "
            "request does not ask to change.\n"
        )

    return (
        f"1) GOAL: {task}\n"
        f"   Format: {meta.format}. Target length: ~{target_seconds}s vertical short.\n"
        "\n"
        "2) FIXED VIRAL ARC (structural contract - every short follows this shape):\n"
        "   hook (2-3s, cold-open visual) -> problem/promise -> 3-4 feature chapters "
        "(2-3 scenes each, building on each other) -> payoff + CTA.\n"
        "\n"
        f"3) BOARD STATUS (session {meta.session_id}, asset {asset_id}):\n"
        f"   Scene reviews: {reviewed}/{len(expected_scenes)} expected scenes reviewed.\n"
        f"{artifact_lines}\n"
        f"   Resume point: {resume_point}\n"
        "   Artifacts already on the board are DONE - do not redo them; continue at the "
        "resume point.\n"
        "\n"
        "4) MANDATORY ORDER: reviews -> storyline -> script -> voice+cutlist -> contact sheet "
        "(save_contact_sheet: ALWAYS right after build_cutlist and BEFORE render_production, "
        "and again after every cutlist rebuild - a cutlist save archives the sheet) -> render "
        "(coding_agent) -> qa. Do not skip or reorder a stage.\n"
        "   CONTACT-SHEET CHECKPOINT (known pattern, no extra session state): the user steers "
        "around the Kontaktbogen purely by follow-up messages. When the task or a user message "
        "says to stop at the contact sheet (e.g. 'bau bis zum Kontaktbogen, dann stopp'), END "
        "the run right after save_contact_sheet and report the sheet's tiles instead of "
        "rendering; a later message (e.g. 'render jetzt') resumes at render_production through "
        "the normal resume flow.\n"
        "\n"
        "5) LANGUAGE + CHARTER: the script MUST be written in German (the source video's "
        "language) - never switch languages mid-script. Coding-agent charter: if voice_fits "
        "comes back False, never shorten the voice - rebuild the cutlist with a longer "
        "per-chapter time budget and render again.\n"
        "\n"
        "6) QA LIMIT: after ONE revise verdict, at most one revision round is allowed - if the "
        "next QA pass still finds issues, deliver anyway with the findings recorded as "
        "warnings instead of looping again.\n"
        f"{follow_up}"
    )


def _parse_outcome(board: Board, result: Any, *, stage: Stage) -> StageOutcome:
    """Read a finished production-team run into an outcome.

    ``weak`` comes from the board's QA verdict, not message scanning: v2's QA reviewer writes a
    structured ``QaReport`` (``verdict="ship"|"revise"``) to the board via ``save_qa_report``
    rather than saying the word "weak" in chat (v1's convention), so a missing report or a
    "revise" verdict is weak and only an explicit "ship" verdict is not.
    """
    text = ""
    for msg in getattr(result, "messages", None) or []:
        to_text = getattr(msg, "to_model_text", None)
        text += (to_text() if callable(to_text) else str(getattr(msg, "content", ""))) + "\n"
    qa = board.load("qa_report")
    weak = not (isinstance(qa, QaReport) and qa.verdict == "ship")
    return StageOutcome(
        status="ok", weak=weak, summary=text.strip()[:2000], team="magentic", stage=stage
    )


def _make_default_execute(board: Board, asset_id: str, deps: ProductionDeps | None) -> ExecuteFn:
    """The default ``ExecuteFn``: lazily builds and runs the real production team.

    Mirrors :func:`orchestrator._default_execute` (build the team, ``asyncio.run`` its ``run``,
    any exception is a hard fail), but the v2 team is board-bound
    (:func:`production_agents.build_production_team` needs ``board``/``asset_id``/``deps``), so
    those are captured in this closure instead of being ``ExecuteFn`` parameters.
    """

    def execute(
        db: Database, config: AgentConfig, stage: Stage, kind: str, task: str
    ) -> StageOutcome:
        import asyncio  # local: only the real run needs the event loop

        try:
            team = build_production_team(
                db, board, config, asset_id=asset_id, stage=stage, deps=deps
            )
            result = asyncio.run(team.run(task=task))
        except Exception as exc:
            logger.warning("magentic production team failed at stage %s: %s", stage, exc)
            return StageOutcome(
                status="hard_fail", weak=False, summary=str(exc), team="magentic", stage=stage
            )
        return _parse_outcome(board, result, stage=stage)

    return execute


def run_production(
    db: Database,
    config: AgentConfig,
    *,
    asset_id: str,
    session_id: str,
    task: str,
    target_seconds: int = 60,
    message: str | None = None,
    execute: ExecuteFn | None = None,
    deps: ProductionDeps | None = None,
) -> dict[str, Any]:
    """Run the v2 production team for one session: resume-aware board + magentic-only A/B ladder.

    A missing asset — or one whose project no longer exists (an orphaned asset) — is reported
    the same way a failed run is, never raised. Otherwise the session's board is opened if it
    already exists, or created fresh (this is what makes a second call for the same
    ``session_id`` a resume rather than a restart - nothing already on the board is touched or
    re-created). ``message`` is a follow-up request on top of an already-produced board (e.g.
    "go back to the previous storyline" or "make the hook punchier"); it assumes a prior
    production, so if the board does not exist yet, this is an error - report it and return
    WITHOUT creating a board (unlike a plain fresh run, a follow-up with no board to follow up
    on is not a valid restart). Stage A runs first; only a ``hard_fail`` escalates to Stage B
    (both magentic-only - v2 has no GraphFlow fallback). Every stage call goes through
    :func:`orchestrator._safe_execute`, so a raising ``execute`` never propagates out of here.
    """
    if repos.get_asset(db, asset_id) is None:
        return {"ok": False, "error": "asset not found", "session_id": session_id}

    try:
        root = board_root_for(db, asset_id, session_id)
    except ValueError:
        # The asset row exists but its project has been deleted out from under it —
        # board_root_for raises (it is a pure/raising path helper); this is the one caller that
        # must never propagate that, so it is turned into the same kind of reported failure as
        # a missing asset instead.
        return {
            "ok": False,
            "error": "project not found",
            "asset_id": asset_id,
            "session_id": session_id,
        }
    try:
        board = Board.open(root)
    except FileNotFoundError:
        if message:
            return {
                "ok": False,
                "error": "unknown session (no board)",
                "asset_id": asset_id,
                "session_id": session_id,
            }
        meta = BoardMeta(
            session_id=session_id,
            asset_id=asset_id,
            created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
            task=task,
            target_seconds=float(target_seconds),
        )
        board = Board.create(root, meta)

    task_text = build_production_task(
        db, board, asset_id=asset_id, task=task, target_seconds=target_seconds, message=message
    )
    run: ExecuteFn = execute if execute is not None else _make_default_execute(
        board, asset_id, deps
    )

    outcome = _safe_execute(run, db, config, "A", "magentic", task_text)
    escalated = False
    if outcome.status == "hard_fail":
        outcome = _safe_execute(run, db, config, "B", "magentic", task_text)
        escalated = True

    expected_scenes = _expected_scene_numbers(db, asset_id)
    render_report = board.load("render_report")
    export_id = render_report.export_id if isinstance(render_report, RenderReport) else None

    return {
        "ok": outcome.status == "ok",
        "status": outcome.status,
        "stage": outcome.stage,
        "team": outcome.team,
        "weak": outcome.weak,
        "escalated": escalated,
        "summary": outcome.summary,
        "session_id": session_id,
        "board": board.status(),
        "export_id": export_id,
        "resume_point": board.resume_point(expected_scenes),
    }
