"""production_orchestrator: board lifecycle, resume-aware task contract, magentic-only ladder,
run entrypoint (Slice 3, Task 8).

DB/asset fixture mirrors ``tests/test_production_tools_review.py``'s ``_seed_scene`` (project +
asset + succeeded analysis run + transcript + a hand-built ONE-scene rough cut via
``created_from=asset_id``) so this file stays self-contained.

The escalation-ladder tests mirror ``tests/test_short_creator_orchestrator.py``'s scripted-
``ExecuteFn`` pattern (no autogen, no LLM) but keyed by stage only: v2 has no GraphFlow fallback,
so ``kind`` is always ``"magentic"`` here. The default executor (build + run the real production
team via ``asyncio.run``) is manual-to-verify, same as v1's ``orchestrator._default_execute``.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from laura.config import Settings
from laura.db import repos
from laura.db.database import Database, SqliteDatabase
from laura.short_creator import orchestrator, production_orchestrator, providers
from laura.short_creator.board import Board
from laura.short_creator.board_models import (
    BestWindow,
    BoardMeta,
    Chapter,
    RenderReport,
    SceneReview,
    Storyline,
)

FPS = 30
SCENE_FRAMES = 150  # 150 frames @ 30fps = 5.0s

# script maps stage -> ("ok"|"hard_fail", weak) or an Exception to raise. Keyed by stage only
# (not (stage, kind), unlike v1) because v2's ladder only ever calls team="magentic".
_Script = dict[str, "tuple[str, bool] | Exception"]


def _seed_scene(tmp_path: Path) -> tuple[Database, str]:
    """Project + asset + succeeded analysis run w/ transcript + a ONE-scene rough cut.

    Returns ``(db, asset_id)``. Mirrors ``test_production_tools_review.py``'s ``_seed_scene``.
    """
    settings = Settings(workspace_root=tmp_path / "ws", start_runner=False)
    db: Database = SqliteDatabase(settings.db_path)
    db.migrate()
    project = repos.create_project(
        db,
        name="p",
        rate_num=FPS,
        rate_den=1,
        drop_frame=False,
        workspace_root=str(tmp_path / "ws" / "proj"),
    )
    asset = repos.create_asset(
        db,
        project_id=project["id"],
        type="video",
        display_name="a.mp4",
        source_path=str(tmp_path / "a.mp4"),
    )
    run = repos.create_analysis_run(db, asset_id=asset["id"], pipeline_version="t", config={})
    repos.start_analysis_run(db, run["id"])
    repos.insert_segment_with_words(
        db,
        asset_id=asset["id"],
        run_id=run["id"],
        speaker_id=None,
        segment={
            "start_sample": 0,
            "end_sample": 96_000,
            "start_frame": 0,
            "end_frame": SCENE_FRAMES,
            "text": "hallo welt schauen wir uns das dashboard an",
            "confidence": 1.0,
        },
        words=[],
    )
    repos.finish_analysis_run(db, run["id"], status="succeeded", diagnostics={})
    timeline = repos.create_timeline(
        db,
        project_id=project["id"],
        name="Rough Cut",
        kind="rough_cut",
        created_from=asset["id"],
    )
    repos.add_timeline_clip(
        db,
        timeline_id=timeline["id"],
        asset_id=asset["id"],
        src_in_frame=0,
        src_out_frame_exclusive=SCENE_FRAMES,
        seq_in_frame=0,
        seq_out_frame_exclusive=SCENE_FRAMES,
        lane=0,
        role="base",
    )
    repos.replace_scenes(db, project["id"], timeline["id"], [(0, SCENE_FRAMES)])
    return db, str(asset["id"])


def _review(scene_number: int = 1) -> SceneReview:
    return SceneReview(
        scene_number=scene_number,
        src_start_frame=0,
        src_end_frame_exclusive=SCENE_FRAMES,
        description="dashboard",
        whats_happening="scrolls",
        hook_score=7,
        best_window=BestWindow(offset_s=0.0, duration_s=3.0),
    )


def _make_execute(script: _Script) -> tuple[orchestrator.ExecuteFn, list[tuple[str, str]]]:
    calls: list[tuple[str, str]] = []

    def execute(
        db: Database,
        config: providers.AgentConfig,
        stage: str,
        kind: str,
        task: str,
    ) -> orchestrator.StageOutcome:
        calls.append((stage, kind))
        assert kind == "magentic"  # v2 has no graph fallback — always "magentic"
        spec = script[stage]
        if isinstance(spec, Exception):
            raise spec
        status, weak = spec
        return orchestrator.StageOutcome(
            status=cast(orchestrator.Status, status),
            weak=weak,
            summary="done",
            team=cast(orchestrator.TeamKind, kind),
            stage=cast(providers.Stage, stage),
        )

    return execute, calls


# --- board_root_for ------------------------------------------------------------------------


def test_board_root_under_workspace(tmp_path: Path) -> None:
    db, asset_id = _seed_scene(tmp_path)

    root = production_orchestrator.board_root_for(db, asset_id, "sess1")

    assert root == tmp_path / "ws" / "proj" / "agent-runs" / "sess1" / "board"
    assert not root.exists()  # pure path construction — no filesystem side effect


# --- build_production_task ------------------------------------------------------------------


def test_task_text_contains_contract_and_resume(tmp_path: Path) -> None:
    db, asset_id = _seed_scene(tmp_path)
    root = production_orchestrator.board_root_for(db, asset_id, "sess1")
    meta = BoardMeta(
        session_id="sess1",
        asset_id=asset_id,
        created_utc="2026-07-13T00:00:00+00:00",
        task="overview short",
        target_seconds=20.0,
    )
    board = Board.create(root, meta)

    fresh = production_orchestrator.build_production_task(
        db, board, asset_id=asset_id, task="overview short", target_seconds=20
    )
    assert "viral arc" in fresh.lower()
    assert "do not redo" in fresh.lower()
    assert "scene_reviews:1" in fresh  # fresh board -> scene 1 not reviewed yet
    # The contact-sheet checkpoint is part of the mandatory order AND documented as the known
    # steer-by-message pattern (stop at the Kontaktbogen / render later) — no session state.
    assert "save_contact_sheet" in fresh
    assert "Kontaktbogen" in fresh
    assert "dann stopp" in fresh
    assert "render jetzt" in fresh

    board.save_scene_review(_review(1))
    board.save(
        "storyline",
        Storyline(
            red_thread="thread",
            arc=[
                Chapter(
                    chapter=1, role="hook", message="m", scene_numbers=[1], target_seconds=3.0
                )
            ],
        ),
    )

    resumed = production_orchestrator.build_production_task(
        db, board, asset_id=asset_id, task="overview short", target_seconds=20
    )
    assert "storyline: DONE" in resumed
    assert "Resume point: script" in resumed


def test_task_text_follow_up_block_only_with_message(tmp_path: Path) -> None:
    """The USER FOLLOW-UP REQUEST section (+ its revert/re-save/never-redo instructions) only
    appears when ``message`` is passed; the board-status block also grows an ``[archived: ...]``
    suffix per artifact, but only for artifacts that actually have an archived version, and only
    while a message is present (a fresh, non-follow-up call stays exactly as before)."""
    db, asset_id = _seed_scene(tmp_path)
    root = production_orchestrator.board_root_for(db, asset_id, "sess1")
    meta = BoardMeta(
        session_id="sess1",
        asset_id=asset_id,
        created_utc="2026-07-13T00:00:00+00:00",
        task="overview short",
        target_seconds=20.0,
    )
    board = Board.create(root, meta)
    chapter = Chapter(chapter=1, role="hook", message="m", scene_numbers=[1], target_seconds=3.0)
    board.save("storyline", Storyline(red_thread="thread v1", arc=[chapter]))
    board.save("storyline", Storyline(red_thread="thread v2", arc=[chapter]))  # -> v2, archives v1

    without_message = production_orchestrator.build_production_task(
        db, board, asset_id=asset_id, task="overview short", target_seconds=20
    )
    assert "USER FOLLOW-UP REQUEST" not in without_message
    assert "[archived:" not in without_message

    with_message = production_orchestrator.build_production_task(
        db,
        board,
        asset_id=asset_id,
        task="overview short",
        target_seconds=20,
        message="make the hook punchier",
    )
    assert "USER FOLLOW-UP REQUEST" in with_message
    assert "make the hook punchier" in with_message
    assert "revert_artifact" in with_message
    assert "archived_versions" in with_message
    assert "highest affected" in with_message.lower()
    assert "never redo" in with_message.lower()
    assert "intact upstream" in with_message.lower()
    assert "storyline: DONE (v2) [archived: v1]" in with_message


# --- run_production -----------------------------------------------------------------------


def test_run_production_creates_board_and_reports(tmp_path: Path) -> None:
    db, asset_id = _seed_scene(tmp_path)
    config = providers.resolve_from_env({})

    # Asset missing is checked before anything else touches the board.
    missing = production_orchestrator.run_production(
        db, config, asset_id="does-not-exist", session_id="sess1", task="t"
    )
    assert missing == {"ok": False, "error": "asset not found", "session_id": "sess1"}

    execute, calls = _make_execute({"A": ("ok", False)})
    result = production_orchestrator.run_production(
        db,
        config,
        asset_id=asset_id,
        session_id="sess1",
        task="overview short",
        target_seconds=20,
        execute=execute,
    )

    assert result["ok"] is True
    assert result["stage"] == "A"
    assert result["team"] == "magentic"
    assert result["escalated"] is False
    assert result["session_id"] == "sess1"
    assert result["export_id"] is None
    assert result["resume_point"]  # non-empty
    assert result["board"]["meta"]["session_id"] == "sess1"
    assert calls == [("A", "magentic")]
    root = production_orchestrator.board_root_for(db, asset_id, "sess1")
    assert (root / "meta.json").is_file()


def test_run_production_message_on_fresh_board_is_error(tmp_path: Path) -> None:
    """A follow-up ``message`` assumes a prior production already exists on the board. A
    session that has never been run has no board yet, so this is reported as an error - and,
    unlike a plain fresh run (no message), no board directory gets created for it."""
    db, asset_id = _seed_scene(tmp_path)
    config = providers.resolve_from_env({})
    execute, calls = _make_execute({})  # empty script — must never be reached

    result = production_orchestrator.run_production(
        db,
        config,
        asset_id=asset_id,
        session_id="sess1",
        task="overview short",
        message="go back to the previous storyline",
        execute=execute,
    )

    assert result == {
        "ok": False,
        "error": "unknown session (no board)",
        "asset_id": asset_id,
        "session_id": "sess1",
    }
    assert calls == []  # never reached team execution
    root = production_orchestrator.board_root_for(db, asset_id, "sess1")
    assert not root.exists()


def test_run_production_message_run_reaches_team_with_follow_up_text(tmp_path: Path) -> None:
    """A follow-up ``message`` against an EXISTING board runs normally, and the user's text
    ends up verbatim in the task text handed to the team (captured via a fake execute)."""
    db, asset_id = _seed_scene(tmp_path)
    root = production_orchestrator.board_root_for(db, asset_id, "sess1")
    meta = BoardMeta(
        session_id="sess1",
        asset_id=asset_id,
        created_utc="2026-01-01T00:00:00+00:00",
        task="original task",
        target_seconds=20.0,
    )
    Board.create(root, meta).save_scene_review(_review(1))
    config = providers.resolve_from_env({})
    captured_tasks: list[str] = []

    def execute(
        db: Database,
        config: providers.AgentConfig,
        stage: str,
        kind: str,
        task: str,
    ) -> orchestrator.StageOutcome:
        captured_tasks.append(task)
        return orchestrator.StageOutcome(
            status="ok",
            weak=False,
            summary="done",
            team=cast(orchestrator.TeamKind, kind),
            stage=cast(providers.Stage, stage),
        )

    result = production_orchestrator.run_production(
        db,
        config,
        asset_id=asset_id,
        session_id="sess1",
        task="overview short",
        message="swap the hook for the dashboard scene",
        execute=execute,
    )

    assert result["ok"] is True
    assert len(captured_tasks) == 1
    assert "swap the hook for the dashboard scene" in captured_tasks[0]
    assert "USER FOLLOW-UP REQUEST" in captured_tasks[0]


def test_run_production_orphaned_asset_never_raises(tmp_path: Path) -> None:
    """An asset row whose project has been deleted out from under it (board_root_for's own
    ValueError, per its docstring) must be reported the same way a missing asset is, not
    raised. FK-enforced cascade deletes normally remove the asset along with its project (see
    repos.delete_project), so this orphan state is reproduced directly on the fixture, the way
    it could arise from any out-of-band data repair that skips the ORM/repo layer."""
    db, asset_id = _seed_scene(tmp_path)
    with db.connection() as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute(
            "UPDATE media_assets SET project_id = ? WHERE id = ?", ("does-not-exist", asset_id)
        )
    config = providers.resolve_from_env({})
    execute, calls = _make_execute({"A": ("ok", False)})

    result = production_orchestrator.run_production(
        db, config, asset_id=asset_id, session_id="sess1", task="t", execute=execute
    )

    assert result == {
        "ok": False,
        "error": "project not found",
        "asset_id": asset_id,
        "session_id": "sess1",
    }
    assert calls == []  # never reached team execution


def test_run_production_reopens_existing_board(tmp_path: Path) -> None:
    db, asset_id = _seed_scene(tmp_path)
    root = production_orchestrator.board_root_for(db, asset_id, "sess1")
    meta = BoardMeta(
        session_id="sess1",
        asset_id=asset_id,
        created_utc="2026-01-01T00:00:00+00:00",
        task="original task",
        target_seconds=20.0,
    )
    Board.create(root, meta).save_scene_review(_review(1))

    config = providers.resolve_from_env({})
    execute, _calls = _make_execute({"A": ("ok", False)})
    result = production_orchestrator.run_production(
        db,
        config,
        asset_id=asset_id,
        session_id="sess1",
        task="a different task text",
        target_seconds=99,
        execute=execute,
    )

    assert result["board"]["meta"]["created_utc"] == "2026-01-01T00:00:00+00:00"
    assert result["board"]["meta"]["task"] == "original task"
    assert result["board"]["scene_reviews"]["count"] == 1
    reopened = Board.open(root)
    assert reopened.scene_reviews()[0].version == 1  # not overwritten


def test_run_production_escalates_on_hard_fail(tmp_path: Path) -> None:
    db, asset_id = _seed_scene(tmp_path)
    config = providers.resolve_from_env({})
    execute, calls = _make_execute({"A": ("hard_fail", False), "B": ("ok", False)})

    result = production_orchestrator.run_production(
        db, config, asset_id=asset_id, session_id="sess1", task="t", execute=execute
    )

    assert result["ok"] is True
    assert result["stage"] == "B"
    assert result["escalated"] is True
    assert calls == [("A", "magentic"), ("B", "magentic")]


def test_run_production_export_id_from_render_report(tmp_path: Path) -> None:
    db, asset_id = _seed_scene(tmp_path)
    root = production_orchestrator.board_root_for(db, asset_id, "sess1")
    meta = BoardMeta(
        session_id="sess1",
        asset_id=asset_id,
        created_utc="2026-01-01T00:00:00+00:00",
        task="t",
        target_seconds=20.0,
    )
    board = Board.create(root, meta)
    board.save(
        "render_report",
        RenderReport(export_id="exp123", video_s=12.0, width=1080, height=1920, checks=[]),
    )

    config = providers.resolve_from_env({})
    execute, _calls = _make_execute({"A": ("ok", False)})
    result = production_orchestrator.run_production(
        db, config, asset_id=asset_id, session_id="sess1", task="t", execute=execute
    )

    assert result["export_id"] == "exp123"


def test_run_production_never_raises(tmp_path: Path) -> None:
    db, asset_id = _seed_scene(tmp_path)
    config = providers.resolve_from_env({})
    execute, calls = _make_execute({"A": RuntimeError("boom"), "B": RuntimeError("boom again")})

    result = production_orchestrator.run_production(
        db, config, asset_id=asset_id, session_id="sess1", task="t", execute=execute
    )

    assert result["ok"] is False
    assert result["status"] == "hard_fail"
    assert result["escalated"] is True
    assert calls == [("A", "magentic"), ("B", "magentic")]
